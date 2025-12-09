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
    Long-running worker that:
      - polls bots_config for all rows with is_running = true
      - for each (user_id, mode) ensures a trading task exists
      - gets per-user Alpaca keys from alpaca_keys
      - decrypts secrets with AES-256-GCM
      - uses AI tickers if available; otherwise marks bot as idle/waiting
      - stops tasks when is_running becomes false

    This loop MUST NEVER exit on normal errors so Render thinks it's healthy.
    """
    logger.info("🔁 AIBOTIX worker loop started (multi-user, multi-bot)")

    # task_key -> asyncio.Task
    bot_tasks: Dict[str, asyncio.Task] = {}

    async def ensure_bot_task(bot_row: Dict[str, Any]) -> None:
        """
        Ensure there is a running trading task for this bot (user_id+mode),
        but only if AI tickers exist. If no AI tickers yet, we mark the bot
        as idle/waiting and do NOT start trade_loop_async yet.
        """
        user_id_raw = bot_row.get("user_id")
        if user_id_raw is None:
            logger.warning("bots_config row missing user_id: %s", bot_row)
            return

        # In Supabase, user_id is uuid; we always treat as string in Python.
        user_id = str(user_id_raw)
        mode = bot_row.get("mode", "paper")
        strategy_id = bot_row.get("strategy_id", "default_rsi")

        task_key = f"{user_id}:{mode}"

        existing = bot_tasks.get(task_key)
        if existing and not existing.done():
            # Task already running for this user+mode
            return

        # --- AI Ticker Auto‑Generation Step ---
        # Attempt to generate AI tickers if none exist yet
        existing_ai = await fetch_ai_tickers(user_id, mode)
        if not existing_ai:
            try:
                raw_scan = await stage_a_screen_and_collect(mode=mode)
                scored = score_tickers(raw_scan, mode=mode)
                await asyncio.to_thread(save_ai_tickers, user_id, mode, scored)
                logger.info("AI tickers generated for user_id=%s mode=%s: %s", user_id, mode, scored)
                
                # Force refresh to start bot instantly
                await asyncio.sleep(0.2)
                ai_tickers = await fetch_ai_tickers(user_id, mode)
            except Exception as e:
                logger.error("AI selector failed for user_id=%s mode=%s: %s", user_id, mode, e)
        
        # Load AI tickers using the new get_top_tickers function
        ticker_list = await asyncio.to_thread(
            get_top_tickers,
            5,
            user_id,
            mode
        )
        
        ai_tickers = ticker_list

        if not ai_tickers:
            # Bot is configured to run, but AI hasn't chosen tickers yet.
            msg = "Bot started, waiting for AI to choose the best tickers to trade"
            logger.info("%s (user_id=%s, mode=%s)", msg, user_id, mode)

            # Clear previous errors
            await clear_bot_error(user_id, mode)

            # Mark bot as idle/waiting in runtime
            await upsert_runtime(
                user_id,
                mode,
                {
                    "last_heartbeat": utc_now_iso(),
                    "status": "idle_waiting",
                    "message": msg,
                    "last_error": None,
                    "last_shutdown": None,
                },
            )

            # Log for UI
            await log_activity(user_id, mode, msg)

            # Do NOT start trade_loop_async yet; we will re-check AI tickers
            # on the next poll iteration.
            return

        # If we reach here, AI tickers exist – we can start the trading bot.
        async def run_bot_task() -> None:
            # 1) Fetch API keys for this user+mode
            creds = await fetch_alpaca_keys(user_id, mode)
            if not creds:
                msg = "Missing Alpaca API keys; please configure credentials"
                logger.error("%s (user_id=%s, mode=%s)", msg, user_id, mode)
                await update_bot_error(user_id, mode, msg)
                # Reflect error in runtime as well
                await mark_bot_stopped(user_id, mode, error=msg, message=msg)
                return

            api_key = creds["api_key"]
            api_secret_enc = creds["api_secret_enc"]
            api_secret = decrypt_secret(api_secret_enc)

            if not api_secret:
                msg = "Failed to decrypt Alpaca API secret"
                logger.error("%s (user_id=%s, mode=%s)", msg, user_id, mode)
                await update_bot_error(user_id, mode, msg)
                await mark_bot_stopped(user_id, mode, error=msg, message=msg)
                return

            # Clear previous config error and mark status as running
            await clear_bot_error(user_id, mode)
            await mark_bot_running(user_id, mode)
            await log_activity(
                user_id,
                mode,
                f"Bot starting (strategy={strategy_id}, tickers={','.join(ai_tickers)})",
            )

            try:
                logger.info(
                    "▶️ Starting bot for user_id=%s, mode=%s, strategy=%s, tickers=%s",
                    user_id,
                    mode,
                    strategy_id,
                    ai_tickers,
                )
                # Initialize trading client with session data
                from bot.aibotix_trading_bot import init_trading_client

                init_trading_client(
                    api_key=api_key,
                    api_secret=api_secret,
                    paper=(mode == "paper"),
                    user_id=user_id,
                    mode=mode,
                )

                # Long-lived trading loop:
                await trade_loop_async(allowed_tickers=ai_tickers)

                # If trade_loop_async ever returns normally, mark as stopped (no error)
                await log_activity(user_id, mode, "trade_loop_async finished normally")
                await mark_bot_stopped(
                    user_id,
                    mode,
                    message="trade_loop_async finished normally",
                )
                logger.info(
                    "✅ trade_loop_async finished for user_id=%s, mode=%s (strategy=%s)",
                    user_id,
                    mode,
                    strategy_id,
                )
            except asyncio.CancelledError:
                await log_activity(user_id, mode, "Bot task cancelled")
                # Mark as stopped without an error – user likely toggled is_running off
                await mark_bot_stopped(
                    user_id,
                    mode,
                    message="Bot task cancelled via is_running toggle",
                )
                logger.info("⏹ Bot task cancelled for %s", task_key)
                raise
            except Exception as e:
                crash_msg = f"Bot crashed: {e!r}"
                await log_activity(user_id, mode, crash_msg)
                logger.exception(
                    "Bot crashed for %s (user_id=%s, mode=%s): %s",
                    task_key,
                    user_id,
                    mode,
                    e,
                )
                await update_bot_error(user_id, mode, crash_msg)
                # Mark shutdown and persist error for UI indicator
                try:
                    await mark_bot_stopped(user_id, mode, error=crash_msg, message=crash_msg)
                except Exception:
                    logger.exception(
                        "Failed to mark shutdown in bot_runtime for user_id=%s mode=%s",
                        user_id,
                        mode,
                    )

        # Create and track the new task
        bot_tasks[task_key] = asyncio.create_task(run_bot_task())

    # Main supervisor loop – NEVER exits on normal errors
    while True:
        try:
            active_rows = await fetch_active_bots()

            # Equity snapshot throttling
            now_ts = datetime.utcnow().timestamp()
            if not hasattr(worker_loop, "_last_equity_save"):
                worker_loop._last_equity_save = 0

            should_save_equity = (now_ts - worker_loop._last_equity_save) >= 300

            # Update heartbeat for all active bots in bot_runtime,
            # whether or not AI tickers exist yet.
            for row in active_rows:
                uid_raw = row.get("user_id")
                if uid_raw is None:
                    continue
                uid = str(uid_raw)
                mode = row.get("mode", "paper")
                await update_heartbeat(uid, mode)

                # Fetch account & save equity every 5 minutes
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
                    else:
                        # No client yet (likely bot hasn't initialized) — skip silently
                        pass

            # Build the set of keys that *should* be running right now
            active_keys = {
                f"{str(r.get('user_id'))}:{r.get('mode', 'paper')}"
                for r in active_rows
                if r.get("user_id") is not None
            }

            # Ensure tasks exist for each active bot (if AI tickers exist)
            for row in active_rows:
                await ensure_bot_task(row)

            # Cancel tasks that are no longer active in bots_config
            for key, task in list(bot_tasks.items()):
                if key not in active_keys:
                    logger.info("🛑 Stopping bot for %s (is_running flag turned off)", key)
                    uid, mode = key.split(":", 1)
                    await log_activity(uid, mode, "Bot stopping due to is_running=False")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    del bot_tasks[key]

            if should_save_equity:
                worker_loop._last_equity_save = now_ts

        except Exception:
            # We NEVER let an exception break the worker – just log and continue
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