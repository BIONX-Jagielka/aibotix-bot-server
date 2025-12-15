import asyncio
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

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
# Basic logging
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("aibotix.worker")


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


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
                }
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
        return supabase.table("bots_config").upsert(config_payload).execute()
    
    try:
        await _run_supabase(_update_config)
        await upsert_runtime(user_id, mode, fields)
    except Exception:
        logger.exception("Failed to update bot status for user_id=%s mode=%s", user_id, mode)
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
                }
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


async def fetch_ai_tickers(user_id: str, mode: str) -> Optional[list[str]]:
    """
    Safe loader for AI tickers. Returns None when 0 rows exist (first‑run),
    instead of treating it as an error. Prevents PGRST116 exceptions.
    """
    def _query():
        return (
            supabase.table("ai_tickers")
            .select("tickers")
            .eq("user_id", user_id)
            .eq("mode", mode)
            .limit(1)
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

        row = data[0] if isinstance(data, list) else data
        tickers = row.get("tickers") or []

        if not tickers:
            logger.info(
                "AI tickers row exists but empty for user_id=%s mode=%s; waiting.",
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
# Worker: bot task lifecycle
# ----------------------------------------
async def worker_loop(poll_interval: int = 10) -> None:
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
                
                # Get AI tickers
                ai_tickers = await asyncio.to_thread(get_top_tickers, 5, user_id, mode)
                if ai_tickers:
                    logger.info("Bot %s using AI tickers: %s", key, ai_tickers)
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
                # Always clean up from registry
                if key in ACTIVE_BOTS:
                    del ACTIVE_BOTS[key]
                    logger.info("Cleaned up task from ACTIVE_BOTS registry: %s", key)
        
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
        
        # Cancel task if it exists
        if key in ACTIVE_BOTS:
            task = ACTIVE_BOTS[key]
            if not task.done():
                logger.info("STOP cancelling running task for %s", key)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Remove from registry
            del ACTIVE_BOTS[key]
            logger.info("STOP removed task from registry: %s", key)
            await log_activity(user_id, mode, "Task cancelled and cleaned from registry")
        else:
            logger.info("STOP cleaning orphaned state for %s (no active task)", key)
            await log_activity(user_id, mode, "Stop request processed - cleaning any orphaned state")
        
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
            
            # Update heartbeats for running bots
            now_ts = datetime.utcnow().timestamp()
            if not hasattr(worker_loop, "_last_equity_save"):
                worker_loop._last_equity_save = 0
            
            should_save_equity = (now_ts - worker_loop._last_equity_save) >= 300
            
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