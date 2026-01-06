import asyncio
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from base64 import b64decode
from supabase import create_client, Client

from bot.aibotix_trading_bot import trade_loop_async, get_trading_client

# --- AI ticker selector imports ---
from bot.ai_ticker_selector_aibotix import (
    stage_a_screen_and_collect,
    score_tickers,
    save_ai_tickers,
    get_top_tickers,
)

# ----------------------------------------
# Global in-memory bot registry
# ----------------------------------------
ACTIVE_BOTS: Dict[str, asyncio.Task] = {}  # key = f"{user_id}:{mode}" -> asyncio.Task

# ----------------------------------------
# AI Ticker Rescanning Configuration
# ----------------------------------------
AI_RESCAN_INTERVAL_SECONDS = 7200  # 2 hours
AI_SCORE_REPLACEMENT_THRESHOLD = 1.25  # 25% stronger score required

# ----------------------------------------
# Worker Heartbeat Configuration
# ----------------------------------------
HEARTBEAT_INTERVAL_SECONDS = 60  # Worker loop heartbeat interval

# ----------------------------------------
# Account Snapshot Configuration
# ----------------------------------------
SNAPSHOT_INTERVAL_SECONDS = 20 * 60  # 20 minutes for trade performance analytics

# Track last AI scan time per user/mode
LAST_AI_SCAN_AT: Dict[str, datetime] = {}  # key = f"{user_id}:{mode}" -> datetime
LAST_MARKET_STATE: Dict[str, bool] = {}  # key = f"{user_id}:{mode}" -> is_open boolean

# ----------------------------------------
# Basic logging
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("aibotix.worker")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------
# Supabase + AES setup
# ----------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set for worker")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ----------------------------------------
# Helper: run blocking Supabase in thread
# ----------------------------------------
async def _run_supabase(fn):
    """Run a blocking Supabase call in a thread so worker stays async."""
    return await asyncio.to_thread(fn)


# ----------------------------------------
# Data access helpers
# ----------------------------------------
async def fetch_active_bots() -> list[Dict[str, Any]]:
    """
    Fetch all bots that should currently be running (multi-user, multi-mode).
    We do NOT crash the worker on Supabase errors – we log and return [].
    """
    def _query():
        return (
            supabase.table("bots_config")
            .select("*")
            .eq("is_running", True)
            .execute()
        )

    try:
        res = await _run_supabase(_query)
        data = getattr(res, "data", None)
        return data or []
    except Exception as e:
        logger.exception("Failed to fetch active bots from bots_config: %s", e)
        return []


async def fetch_alpaca_keys(user_id: str, mode: str) -> Optional[Dict[str, str]]:
    """
    Fetch encrypted Alpaca credentials for a given user+mode from alpaca_keys.
    Returns None if missing or on error.
    """
    def _query():
        return (
            supabase.table("alpaca_keys")
            .select("api_key_id, api_secret_enc")
            .eq("user_id", user_id)
            .eq("mode", mode)
            .single()
            .execute()
        )

    try:
        res = await _run_supabase(_query)
        data = getattr(res, "data", None)
        if not data:
            logger.warning("No Alpaca keys found for user_id=%s, mode=%s", user_id, mode)
            return None
        return {
            "api_key": data["api_key_id"],
            "api_secret_enc": data["api_secret_enc"],
        }
    except Exception as e:
        logger.exception(
            "Failed to fetch Alpaca keys for user_id=%s mode=%s: %s",
            user_id,
            mode,
            e,
        )
        return None


async def update_bot_error(user_id: str, mode: str, error: str) -> None:
    """
    Persist an error for a bot and mark it as not running so the UI can show it.
    """
    def _update():
        return (
            supabase.table("bots_config")
            .update({"is_running": False, "last_error": error})
            .eq("user_id", user_id)
            .eq("mode", mode)
            .execute()
        )

    try:
        await _run_supabase(_update)
    except Exception:
        logger.exception("Failed to update bot error for user_id=%s mode=%s", user_id, mode)


async def clear_bot_error(user_id: str, mode: str) -> None:
    """
    Clear last_error in bots_config when a bot successfully starts or recovers.
    """
    def _update():
        return (
            supabase.table("bots_config")
            .update({"last_error": None})
            .eq("user_id", user_id)
            .eq("mode", mode)
            .execute()
        )

    try:
        await _run_supabase(_update)
    except Exception:
        logger.exception("Failed to clear bot error for user_id=%s mode=%s", user_id, mode)


async def log_activity(user_id: str, mode: str, message: str) -> None:
    """Insert a log entry into bot_logs."""
    def _insert():
        return (
            supabase.table("bot_logs")
            .insert(
                {
                    "user_id": user_id,
                    "mode": mode,
                    "message": message,
                    "created_at": utc_now_iso(),
                }
            )
            .execute()
        )

    try:
        await _run_supabase(_insert)
    except Exception:
        logger.exception(
            "Failed to insert bot_logs entry for user_id=%s, mode=%s: %s",
            user_id,
            mode,
            message,
        )


async def upsert_runtime(user_id: str, mode: str, fields: Dict[str, Any]) -> None:
    """
    Shared helper to upsert into bot_runtime for a given user+mode.
    This table stores live runtime metrics (heartbeat, last error, shutdown, status).
    """
    def _upsert():
        payload = {**fields}
        return (
            supabase.table("bot_runtime")
            .upsert(
                {
                    "user_id": user_id,
                    "mode": mode,
                    **payload,
                },
                on_conflict="user_id,mode"
            )
            .execute()
        )

    try:
        await _run_supabase(_upsert)
    except Exception:
        logger.exception(
            "Failed to upsert bot_runtime for user_id=%s mode=%s with %s",
            user_id,
            mode,
            fields,
        )


async def upsert_bot_status(user_id: str, mode: str, status: str, **kwargs) -> None:
    """
    Helper to update bot status with consistent field names.
    Status values: "running" | "stopped" | "error"
    """
    fields = {
        "status": status,
        "updated_at": utc_now_iso(),
        **kwargs
    }
    
    # Also update bots_config table for consistency
    def _update_config():
        config_payload = {
            "user_id": user_id,
            "mode": mode,
            "is_running": status == "running",
            "updated_at": utc_now_iso(),
        }
        if "last_error" in kwargs:
            config_payload["last_error"] = kwargs["last_error"]
        return supabase.table("bots_config").upsert(config_payload, on_conflict="user_id,mode").execute()
    
    try:
        await _run_supabase(_update_config)
        await upsert_runtime(user_id, mode, fields)
    except Exception:
        logger.exception("Failed to update bot status for user_id=%s mode=%s", user_id, mode)


async def mark_bot_running(user_id: str, mode: str) -> None:
    """
    Mark bot as running by updating last_heartbeat and clearing runtime error/shutdown.
    """
    await upsert_runtime(
        user_id,
        mode,
        {
            "last_heartbeat": utc_now_iso(),
            "last_error": None,
            "last_shutdown": None,
            "status": "running",
            "message": "Bot running normally",
        },
    )


async def mark_bot_stopped(
    user_id: str,
    mode: str,
    *,
    error: Optional[str] = None,
    message: Optional[str] = None,
) -> None:
    """
    Mark bot as stopped. Optionally persist the last error message and custom message.
    """
    fields: Dict[str, Any] = {
        "last_shutdown": utc_now_iso(),
        "status": "stopped",
    }
    if error is not None:
        fields["last_error"] = error
    if message is not None:
        fields["message"] = message

    await upsert_runtime(user_id, mode, fields)


async def update_heartbeat(user_id: str, mode: str) -> None:
    """
    Lightweight heartbeat writer used while the bot is configured as running.
    """
    await upsert_runtime(
        user_id,
        mode,
        {
            "last_heartbeat": utc_now_iso(),
        },
    )


async def is_market_open(trading_client) -> bool:
    """
    Check if market is currently open using the trading client.
    """
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception:
        return False


async def should_trigger_ai_scan(user_id: str, mode: str, trading_client) -> tuple[bool, str]:
    """
    Determine if AI ticker scan should be triggered.
    Returns (should_scan, reason)
    """
    key = f"{user_id}:{mode}"
    now = datetime.now(timezone.utc)
    
    # Check current market state
    current_market_open = await is_market_open(trading_client)
    previous_market_open = LAST_MARKET_STATE.get(key, False)
    
    # Update market state tracking
    LAST_MARKET_STATE[key] = current_market_open
    
    # Trigger 1: Market just opened
    if current_market_open and not previous_market_open:
        return True, "market open"
    
    # Trigger 2: Scheduled rescan (market is open and enough time has passed)
    if current_market_open:
        last_scan = LAST_AI_SCAN_AT.get(key)
        if last_scan is None:
            return True, "initial scan"
        
        time_since_scan = (now - last_scan).total_seconds()
        if time_since_scan >= AI_RESCAN_INTERVAL_SECONDS:
            return True, "scheduled rescan"
    
    return False, "no trigger"


async def intelligent_ticker_replacement(
    user_id: str, mode: str, new_tickers: list, current_ai_tickers: list, trading_client
) -> tuple[list[str], list[str], list[str]]:
    """
    Intelligent ticker replacement logic that preserves profitable positions.
    Returns (final_tickers, kept_tickers, replaced_tickers)
    """
    if not new_tickers:
        return current_ai_tickers, current_ai_tickers, []
    
    if not current_ai_tickers:
        return new_tickers, [], new_tickers
    
    try:
        # Get current positions
        positions = await asyncio.to_thread(trading_client.get_all_positions)
        position_symbols = {p.symbol for p in positions}
        profitable_positions = set()
        
        # Identify profitable positions
        for position in positions:
            try:
                pnl = float(position.unrealized_plpc)
                if pnl >= 0:  # Profitable or break-even
                    profitable_positions.add(position.symbol)
            except (AttributeError, ValueError):
                # If we can't determine PnL, assume it's held for a reason
                profitable_positions.add(position.symbol)
    
    except Exception as e:
        logger.warning("Failed to get positions for ticker replacement: %s", e)
        # If we can't get positions, be conservative and keep current tickers
        return current_ai_tickers, current_ai_tickers, []
    
    # Score-based replacement logic
    final_tickers = []
    kept_tickers = []
    replaced_tickers = []
    
    # Keep profitable existing positions
    for ticker in current_ai_tickers:
        if ticker in position_symbols and ticker in profitable_positions:
            final_tickers.append(ticker)
            kept_tickers.append(ticker)
            logger.info("Keeping profitable position: %s", ticker)
    
    # Fill remaining slots with new tickers if they meet threshold
    new_ticker_set = set(new_tickers) - set(final_tickers)
    remaining_slots = 5 - len(final_tickers)  # Assuming max 5 tickers
    
    # Add new tickers up to the limit
    for ticker in list(new_ticker_set)[:remaining_slots]:
        final_tickers.append(ticker)
        if ticker not in current_ai_tickers:
            replaced_tickers.append(ticker)
    
    # If we still have slots, fill with non-profitable current tickers
    if len(final_tickers) < 5:
        for ticker in current_ai_tickers:
            if ticker not in final_tickers and len(final_tickers) < 5:
                final_tickers.append(ticker)
                kept_tickers.append(ticker)
    
    return final_tickers[:5], kept_tickers, replaced_tickers


async def fetch_ai_tickers(user_id: str, mode: str) -> Optional[list[str]]:
    """
    Safe loader for AI tickers. Returns None when 0 rows exist (first‑run),
    instead of treating it as an error. Prevents PGRST116 exceptions.
    """
    def _query():
        return (
            supabase.table("ai_tickers")
            .select("ticker")
            .eq("user_id", user_id)
            .eq("mode", mode)
            .order("rank")
            .execute()
        )

    try:
        res = await _run_supabase(_query)
        data = getattr(res, "data", None)

        # If no rows exist yet → expected on first boot
        if not data:
            logger.info(
                "No AI tickers found for user_id=%s mode=%s (first‑run or not generated yet).",
                user_id,
                mode,
            )
            return None

        # Build list from individual ticker rows
        tickers = [row["ticker"] for row in data if row.get("ticker")]

        if not tickers:
            logger.info(
                "AI tickers rows exist but empty for user_id=%s mode=%s; waiting.",
                user_id,
                mode,
            )
            return None

        return tickers

    except Exception as e:
        msg = str(e)
        # Safe handling for Supabase "0 rows" error
        if "PGRST116" in msg or "0 rows" in msg:
            logger.info(
                "AI tickers not ready yet for user_id=%s mode=%s (Supabase returned empty).",
                user_id,
                mode,
            )
            return None

        logger.exception(
            "Unexpected error while fetching AI tickers for user_id=%s mode=%s: %s",
            user_id,
            mode,
            e,
        )
        return None


# ----------------------------------------
# Crypto helper
# ----------------------------------------
def decrypt_secret(enc_b64: str) -> Optional[str]:
    """
    Decrypt AES-256-GCM secret produced by Next.js API (save-keys route).
    Layout: [12-byte IV][16-byte TAG][ciphertext] all base64 encoded.
    ENV: ENCRYPTION_KEY must be a 32-byte key in base64 format.
    """
    try:
        if not ENCRYPTION_KEY:
            logger.error("ENCRYPTION_KEY missing — cannot decrypt")
            return None

        key = b64decode(ENCRYPTION_KEY)
        raw = b64decode(enc_b64)

        iv = raw[:12]
        tag = raw[12:28]
        ciphertext = raw[28:]

        decryptor = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
        ).decryptor()

        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext.decode()

    except Exception as e:
        logger.error("AES-GCM decrypt failed: %s", e)
        return None


# ----------------------------------------
# Helper: Save equity snapshot
# ----------------------------------------
async def save_equity_snapshot(supabase, user_id: str, account):
    try:
        equity = float(account.portfolio_value)
        cash = float(account.cash)
        pos_val = float(getattr(account, "position_market_value", 0))

        def _insert():
            return (
                supabase.table("equity_history")
                .insert(
                    {
                        "user_id": user_id,
                        "equity": equity,
                        "cash": cash,
                        "positions_value": pos_val,
                    }
                )
                .execute()
            )

        await asyncio.to_thread(_insert)

    except Exception as e:
        logger.error("Failed to save equity snapshot: %s", e)


# ----------------------------------------
# Helper: Account Performance Snapshot
# ----------------------------------------
async def save_account_snapshot(user_id: str, mode: str, snapshot_reason: str):
    """
    Single source of truth for account snapshots.
    Inserts one row into account_snapshots table for trade performance analytics.
    """
    try:
        # Get trading client for this user+mode
        client_bundle = get_trading_client(user_id, mode)
        if not client_bundle or not client_bundle.get("trading_client"):
            logger.warning(f"[SNAPSHOT] No trading client found for {user_id}:{mode}")
            return
        
        trading_client = client_bundle["trading_client"]
        
        # Fetch Alpaca account data
        account = await asyncio.to_thread(trading_client.get_account)
        positions = await asyncio.to_thread(trading_client.get_all_positions)

        # Extract account metrics
        equity = float(account.equity)
        cash = float(account.cash)
        buying_power = float(account.buying_power)
        
        # Calculate position metrics
        capital_in_use = 0.0
        open_positions_value = 0.0
        unrealized_pnl = 0.0

        for p in positions:
            capital_in_use += float(getattr(p, 'cost_basis', 0) or 0)
            open_positions_value += float(getattr(p, 'market_value', 0) or 0)
            unrealized_pnl += float(getattr(p, 'unrealized_pl', 0) or 0)

        # Calculate P&L metrics
        realized_pnl = float(getattr(account, 'realized_pl', 0) or 0)
        total_pnl = realized_pnl + unrealized_pnl

        # Prepare snapshot data
        payload = {
            "user_id": user_id,
            "mode": mode,
            "equity": equity,
            "cash": cash,
            "buying_power": buying_power,
            "capital_in_use": capital_in_use,
            "open_positions_value": open_positions_value,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "snapshot_reason": snapshot_reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Insert into account_snapshots table (insert-only)
        def _insert():
            return supabase.table("account_snapshots").insert(payload).execute()

        await asyncio.to_thread(_insert)
        logger.debug(f"[SNAPSHOT] {snapshot_reason} snapshot saved for {user_id}:{mode}")

    except Exception as e:
        logger.warning(f"[SNAPSHOT] Failed to save account snapshot: {e}")


# ----------------------------------------
# Worker: bot task lifecycle
# ----------------------------------------
async def worker_loop(poll_interval: int = HEARTBEAT_INTERVAL_SECONDS) -> None:
    """
    Long-running worker with robust bot start/stop state management.
    
    GOAL: Make bot START and STOP idempotent and reliable even if Supabase 
    contains stale records or the worker restarts.
    
    RULES:
    - ACTIVE_BOTS registry is the source of truth for running tasks
    - Supabase represents INTENT, not proof of execution
    - START always works if not already running
    - STOP always works and cleans up state
    - No manual database cleanup required
    """
    logger.info("🔁 AIBOTIX worker loop started (robust state management)")
    logger.info(f"⏰ Worker heartbeat interval: {HEARTBEAT_INTERVAL_SECONDS} seconds")
    
    # On startup, ACTIVE_BOTS starts empty - no auto-start from stale Supabase records
    logger.info("Worker startup: ACTIVE_BOTS registry initialized empty")

    async def start_bot_task(user_id: str, mode: str, strategy_id: str = "default_rsi") -> bool:
        """
        Start a bot task with idempotent logic.
        Returns True if started/already running, False if failed.
        """
        key = f"{user_id}:{mode}"
        
        # Check if task already exists and is running
        if key in ACTIVE_BOTS:
            task = ACTIVE_BOTS[key]
            if not task.done():
                logger.info("START ignored: bot already running for %s", key)
                await log_activity(user_id, mode, "Start request ignored - bot already running")
                return True
            else:
                # Task exists but is done - clean it up
                logger.info("Cleaning up completed task for %s", key)
                del ACTIVE_BOTS[key]
        
        # Check for stale Supabase state and override
        try:
            resp = await _run_supabase(lambda: (
                supabase.table("bot_runtime")
                .select("status")
                .eq("user_id", user_id)
                .eq("mode", mode)
                .limit(1)
                .execute()
            ))
            
            if resp.data and resp.data[0].get("status") == "running":
                logger.info("START overriding stale Supabase state for %s", key)
                await log_activity(user_id, mode, "Overriding stale database state - starting fresh bot")
        except Exception:
            pass  # Don't let status checks block startup
        
        # Fetch API keys
        creds = await fetch_alpaca_keys(user_id, mode)
        if not creds:
            error_msg = "Missing Alpaca API keys; cannot start bot"
            logger.error("%s for %s", error_msg, key)
            await upsert_bot_status(user_id, mode, "error", last_error=error_msg, message=error_msg)
            await log_activity(user_id, mode, f"❌ {error_msg}")
            return False
        
        api_key = creds["api_key"]
        api_secret_enc = creds["api_secret_enc"]
        api_secret = decrypt_secret(api_secret_enc)
        
        if not api_secret:
            error_msg = "Failed to decrypt Alpaca API secret"
            logger.error("%s for %s", error_msg, key)
            await upsert_bot_status(user_id, mode, "error", last_error=error_msg, message=error_msg)
            await log_activity(user_id, mode, f"❌ {error_msg}")
            return False
        
        # Create bot task
        async def run_bot_task() -> None:
            try:
                logger.info("▶️ Starting bot task for %s", key)
                await upsert_bot_status(user_id, mode, "running", 
                                      last_error=None, 
                                      last_heartbeat=utc_now_iso(),
                                      message="Bot running normally")
                await log_activity(user_id, mode, f"✅ Bot started (strategy={strategy_id})")
                
                # Initialize trading client
                from bot.aibotix_trading_bot import init_trading_client
                init_trading_client(
                    api_key=api_key,
                    api_secret=api_secret,
                    paper=(mode == "paper"),
                    user_id=user_id,
                    mode=mode,
                )
                
                # Take initial account snapshot after bot startup
                try:
                    await save_account_snapshot(user_id, mode, "bot_start")
                except Exception as e:
                    logger.warning(f"Failed to take startup snapshot for {key}: {e}")
                
                # Get AI tickers and initialize scan tracking
                ai_tickers = await asyncio.to_thread(get_top_tickers, 5, user_id, mode)
                
                # Initialize AI scan tracking for this bot
                bot_key = f"{user_id}:{mode}"
                LAST_AI_SCAN_AT[bot_key] = datetime.now(timezone.utc)
                
                # Initialize market state (assume closed to trigger immediate scan at open)
                LAST_MARKET_STATE[bot_key] = False
                
                if ai_tickers:
                    logger.info("Bot %s using initial AI tickers: %s", key, ai_tickers)
                    await trade_loop_async(allowed_tickers=ai_tickers)
                else:
                    logger.info("Bot %s starting with no AI tickers - will wait for selection", key)
                    await trade_loop_async()
                
                # Normal completion
                logger.info("✅ Bot task completed normally for %s", key)
                await upsert_bot_status(user_id, mode, "stopped", 
                                      message="Bot completed normally")
                await log_activity(user_id, mode, "Bot completed trading loop normally")
                
            except asyncio.CancelledError:
                logger.info("⏹ Bot task cancelled for %s", key)
                await upsert_bot_status(user_id, mode, "stopped", 
                                      message="Bot stopped via stop request")
                await log_activity(user_id, mode, "🛑 Bot stopped via stop request")
                raise
                
            except Exception as e:
                error_msg = f"Bot crashed: {e!r}"
                logger.exception("Bot crashed for %s: %s", key, e)
                await upsert_bot_status(user_id, mode, "error", 
                                      last_error=error_msg, 
                                      message=error_msg)
                await log_activity(user_id, mode, f"❌ {error_msg}")
            finally:
                # Only remove registry entry if it still points to THIS task (avoids races on restart)
                try:
                    current = ACTIVE_BOTS.get(key)
                    if current is asyncio.current_task():
                        ACTIVE_BOTS.pop(key, None)
                        logger.info("Cleaned up task from ACTIVE_BOTS registry: %s", key)
                except Exception:
                    # Never let cleanup crash the worker
                    pass
                
                # Clean up AI scan tracking
                if key in LAST_AI_SCAN_AT:
                    del LAST_AI_SCAN_AT[key]
                if key in LAST_MARKET_STATE:
                    del LAST_MARKET_STATE[key]
        
        # Create and store the task
        task = asyncio.create_task(run_bot_task())
        ACTIVE_BOTS[key] = task
        
        logger.info("Started new bot task for %s", key)
        return True
    
    async def stop_bot_task(user_id: str, mode: str) -> bool:
        """
        Stop a bot task with idempotent logic.
        Returns True always (STOP never blocks future START calls).
        """
        key = f"{user_id}:{mode}"
        
        # Cancel task if it exists (idempotent, race-safe)
        task = ACTIVE_BOTS.pop(key, None)
        if task is not None:
            if not task.done():
                logger.info("STOP cancelling running task for %s", key)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            logger.info("STOP removed task from registry: %s", key)
            await log_activity(user_id, mode, "Task cancelled and cleaned from registry")
        else:
            logger.info("STOP cleaning orphaned state for %s (no active task)", key)
            await log_activity(user_id, mode, "Stop request processed - cleaning any orphaned state")
        
        # Clean up AI scan tracking
        if key in LAST_AI_SCAN_AT:
            del LAST_AI_SCAN_AT[key]
        if key in LAST_MARKET_STATE:
            del LAST_MARKET_STATE[key]
        
        # Take final account snapshot before stopping
        try:
            await save_account_snapshot(user_id, mode, "bot_stop")
        except Exception as e:
            logger.warning(f"Failed to take shutdown snapshot for {key}: {e}")
        
        # Always update Supabase status to stopped
        await upsert_bot_status(user_id, mode, "stopped", 
                              message="Bot stopped via stop request")
        
        logger.info("STOP completed for %s", key)
        return True

    # Main supervisor loop
    while True:
        try:
            # Fetch bot configuration intent from Supabase
            active_rows = await fetch_active_bots()
            
            # Update heartbeats for running bots and handle AI rescanning
            now_ts = datetime.now(timezone.utc).timestamp()
            if not hasattr(worker_loop, "_last_equity_save"):
                worker_loop._last_equity_save = 0
            if not hasattr(worker_loop, "_last_snapshot_save"):
                worker_loop._last_snapshot_save = 0
            
            should_save_equity = (now_ts - worker_loop._last_equity_save) >= 300
            should_save_snapshot = (now_ts - worker_loop._last_snapshot_save) >= SNAPSHOT_INTERVAL_SECONDS
            
            # AI Ticker Rescanning Logic
            for row in active_rows:
                uid_raw = row.get("user_id")
                if uid_raw is None:
                    continue
                uid = str(uid_raw)
                mode = row.get("mode", "paper")
                key = f"{uid}:{mode}"
                
                # Only process AI rescanning for running bots
                if key in ACTIVE_BOTS and not ACTIVE_BOTS[key].done():
                    try:
                        # Get trading client for market state checks
                        client_bundle = get_trading_client(uid, mode)
                        if client_bundle and client_bundle.get("trading_client"):
                            trading_client = client_bundle["trading_client"]
                            
                            # Check if AI scan should be triggered
                            should_scan, reason = await should_trigger_ai_scan(uid, mode, trading_client)
                            
                            if should_scan:
                                logger.info("AI ticker scan triggered: %s (user_id=%s, mode=%s)", reason, uid, mode)
                                
                                try:
                                    # Get current AI tickers
                                    current_tickers = await fetch_ai_tickers(uid, mode) or []
                                    
                                    # Run AI ticker selection
                                    new_tickers = await asyncio.to_thread(get_top_tickers, 5, uid, mode)
                                    
                                    if new_tickers:
                                        # Intelligent replacement logic
                                        final_tickers, kept_tickers, replaced_tickers = await intelligent_ticker_replacement(
                                            uid, mode, new_tickers, current_tickers, trading_client
                                        )
                                        
                                        # Update AI tickers if changes were made
                                        if final_tickers != current_tickers:
                                            # Save new tickers to database
                                            await asyncio.to_thread(
                                                lambda: supabase.table("ai_tickers")
                                                .delete()
                                                .match({"user_id": uid, "mode": mode})
                                                .execute()
                                            )
                                            
                                            rows = []
                                            for idx, ticker in enumerate(final_tickers, start=1):
                                                rows.append({
                                                    "user_id": uid,
                                                    "mode": mode,
                                                    "ticker": ticker,
                                                    "rank": idx,
                                                    "score": 0.0,  # Will be updated by next full scan
                                                })
                                            
                                            if rows:
                                                await asyncio.to_thread(
                                                    lambda: supabase.table("ai_tickers").insert(rows).execute()
                                                )
                                            
                                            # Log changes
                                            if kept_tickers:
                                                logger.info("AI scan kept profitable tickers: %s", kept_tickers)
                                            if replaced_tickers:
                                                logger.info("AI scan added new opportunities: %s", replaced_tickers)
                                        else:
                                            logger.info("AI scan completed - no changes needed (tickers remain optimal)")
                                    else:
                                        logger.warning("AI scan returned no tickers - keeping current selection")
                                    
                                    # Update last scan timestamp
                                    LAST_AI_SCAN_AT[key] = datetime.now(timezone.utc)
                                    
                                except Exception as scan_error:
                                    logger.error("AI ticker scan failed for %s: %s", key, scan_error)
                                    # Keep current tickers on scan failure - safety fallback
                    
                    except Exception as e:
                        logger.warning("Error in AI rescanning logic for %s: %s", key, e)
            
            for row in active_rows:
                uid_raw = row.get("user_id")
                if uid_raw is None:
                    continue
                uid = str(uid_raw)
                mode = row.get("mode", "paper")
                key = f"{uid}:{mode}"
                
                # Update heartbeat if task is actually running
                if key in ACTIVE_BOTS and not ACTIVE_BOTS[key].done():
                    await update_heartbeat(uid, mode)
                
                # Equity snapshots
                if should_save_equity:
                    client_bundle = get_trading_client(uid, mode)
                    if client_bundle:
                        trading_client = client_bundle.get("trading_client")
                        if trading_client:
                            try:
                                account = trading_client.get_account()
                                await save_equity_snapshot(supabase, uid, account)
                            except Exception as e:
                                logger.error("Failed to fetch account for equity snapshot: %s", e)
                
                # Account performance snapshots (20-minute interval)
                if should_save_snapshot and key in ACTIVE_BOTS and not ACTIVE_BOTS[key].done():
                    await save_account_snapshot(uid, mode, "heartbeat")
            
            # Determine which bots should be running (START logic)
            intended_running = {
                f"{str(r.get('user_id'))}:{r.get('mode', 'paper')}"
                for r in active_rows
                if r.get("user_id") is not None
            }
            
            # Start tasks for bots that should be running but aren't
            for row in active_rows:
                uid_raw = row.get("user_id")
                if uid_raw is None:
                    continue
                uid = str(uid_raw)
                mode = row.get("mode", "paper")
                strategy_id = row.get("strategy_id", "default_rsi")
                key = f"{uid}:{mode}"
                
                if key not in ACTIVE_BOTS or ACTIVE_BOTS[key].done():
                    logger.info("Starting bot for %s (requested via is_running=true)", key)
                    await start_bot_task(uid, mode, strategy_id)
            
            # Stop tasks that are no longer intended to run (STOP logic)
            for key in list(ACTIVE_BOTS.keys()):
                if key not in intended_running:
                    uid, mode = key.split(":", 1)
                    logger.info("Stopping bot for %s (is_running=false detected)", key)
                    await stop_bot_task(uid, mode)
            
            if should_save_equity:
                worker_loop._last_equity_save = now_ts
            if should_save_snapshot:
                worker_loop._last_snapshot_save = now_ts
        
        except Exception:
            logger.exception("Worker loop iteration failed")
        
        await asyncio.sleep(poll_interval)


def main() -> None:
    """
    Entry point when running this module directly as a worker process.
    This keeps running forever until interrupted.
    """
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker interrupted, shutting down")


if __name__ == "__main__":
    main()