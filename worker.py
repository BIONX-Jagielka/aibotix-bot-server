import asyncio
import logging
import os
from typing import Dict, Any, Optional

from cryptography.fernet import Fernet, InvalidToken
from supabase import create_client, Client

from bot.aibotix_trading_bot import trade_loop_async

# ----------------------------------------
# Basic logging
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("aibotix.worker")

# ----------------------------------------
# Supabase + Fernet setup
# ----------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
FERNET_KEY = os.environ.get("FERNET_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set for worker")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

fernet: Optional[Fernet] = None
if FERNET_KEY:
    try:
        fernet = Fernet(FERNET_KEY.encode())
    except Exception as e:
        logger.error("Invalid FERNET_KEY in worker: %s", e)
else:
    logger.warning("FERNET_KEY not set; worker cannot decrypt Alpaca secrets")


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
        logger.exception("Failed to fetch Alpaca keys for user_id=%s mode=%s: %s", user_id, mode, e)
        return None


async def update_bot_error(user_id: str, mode: str, error: str) -> None:
    """
    Persist an error for a bot and mark it as not running so the UI can show it.
    """
    def _update():
        return (
            supabase.table("bots_config")
            .update({"is_running": False, "last_error": error, "updated_at": "now()"})
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
    Optional helper to clear error when bot successfully restarts.
    """
    def _update():
        return (
            supabase.table("bots_config")
            .update({"last_error": None, "updated_at": "now()"})
            .eq("user_id", user_id)
            .eq("mode", mode)
            .execute()
        )

    try:
        await _run_supabase(_update)
    except Exception:
        logger.exception("Failed to clear bot error for user_id=%s mode=%s", user_id, mode)


# ----------------------------------------
# Crypto helper
# ----------------------------------------
def decrypt_secret(enc: str) -> Optional[str]:
    if not enc:
        return None
    if not fernet:
        logger.error("FERNET_KEY not available; cannot decrypt Alpaca secret")
        return None
    try:
        return fernet.decrypt(enc.encode()).decode()
    except InvalidToken:
        logger.error("Invalid Fernet token while decrypting Alpaca secret")
        return None
    except Exception as e:
        logger.exception("Unexpected error decrypting Alpaca secret: %s", e)
        return None


# ----------------------------------------
# Worker loop
# ----------------------------------------
async def worker_loop(poll_interval: int = 10) -> None:
    """
    Long-running worker that:
      - polls bots_config for all rows with is_running = true
      - for each (user_id, mode) ensures a trading task exists
      - gets per-user Alpaca keys from alpaca_keys
      - decrypts secrets with Fernet
      - stops tasks when is_running becomes false
    This loop MUST NEVER exit on normal errors so Render thinks it's healthy.
    """
    logger.info("🔁 AIBOTIX worker loop started (multi-user, multi-bot)")

    # task_key -> asyncio.Task
    bot_tasks: Dict[str, asyncio.Task] = {}

    async def ensure_bot_task(bot_row: Dict[str, Any]) -> None:
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

        async def run_bot_task() -> None:
            # 1) Fetch API keys for this user+mode
            creds = await fetch_alpaca_keys(user_id, mode)
            if not creds:
                msg = "Missing Alpaca API keys; please configure credentials"
                logger.error("%s (user_id=%s, mode=%s)", msg, user_id, mode)
                await update_bot_error(user_id, mode, msg)
                return

            api_key = creds["api_key"]
            api_secret_enc = creds["api_secret_enc"]
            api_secret = decrypt_secret(api_secret_enc)

            if not api_secret:
                msg = "Failed to decrypt Alpaca API secret"
                logger.error("%s (user_id=%s, mode=%s)", msg, user_id, mode)
                await update_bot_error(user_id, mode, msg)
                return

            # Clear last_error on successful startup
            await clear_bot_error(user_id, mode)

            try:
                logger.info(
                    "▶️ Starting bot for user_id=%s, mode=%s, strategy=%s",
                    user_id,
                    mode,
                    strategy_id,
                )
                # This call should be a long-lived loop inside the bot.
                await trade_loop_async(
                    mode=mode,
                    api_key=api_key,
                    api_secret=api_secret,
                    user_id=user_id,
                    strategy_id=strategy_id,
                )
                # If trade_loop_async returns normally, we just log and the worker
                # will decide whether to restart it on the next poll, depending on is_running.
                logger.info(
                    "✅ trade_loop_async finished for user_id=%s, mode=%s (strategy=%s)",
                    user_id,
                    mode,
                    strategy_id,
                )
            except asyncio.CancelledError:
                logger.info("⏹ Bot task cancelled for %s", task_key)
                raise
            except Exception as e:
                logger.exception(
                    "Bot crashed for %s (user_id=%s, mode=%s): %s",
                    task_key,
                    user_id,
                    mode,
                    e,
                )
                await update_bot_error(user_id, mode, f"Bot crashed: {e!r}")

        # Create and track the new task
        bot_tasks[task_key] = asyncio.create_task(run_bot_task())

    # Main supervisor loop – NEVER exits on normal errors
    while True:
        try:
            active_rows = await fetch_active_bots()
            # Build the set of keys that *should* be running right now
            active_keys = {
                f"{str(r.get('user_id'))}:{r.get('mode', 'paper')}"
                for r in active_rows
                if r.get("user_id") is not None
            }

            # Ensure tasks exist for each active bot
            for row in active_rows:
                await ensure_bot_task(row)

            # Cancel tasks that are no longer active
            for key, task in list(bot_tasks.items()):
                if key not in active_keys:
                    logger.info("🛑 Stopping bot for %s (is_running flag turned off)", key)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    del bot_tasks[key]

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