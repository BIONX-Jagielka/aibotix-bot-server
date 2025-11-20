import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request
from supabase import create_client, Client
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------
# Startup: Supabase + Encryption
# --------------------------------------------------------------------
try:
    print("🚀 Starting AIBOTIX bot server...")

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY or not ENCRYPTION_KEY:
        raise EnvironmentError("❌ Missing one or more required environment variables: SUPABASE_URL, SUPABASE_KEY, ENCRYPTION_KEY")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    fernet = Fernet(ENCRYPTION_KEY.encode("utf-8"))

    print("✅ Supabase client and Fernet key initialized successfully.")
except Exception as e:
    import traceback

    print("❌ Fatal error during startup:", e)
    traceback.print_exc()
    # Re-raise so Render fails fast if config is broken
    raise e

app = FastAPI()


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def utc_now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def get_alpaca_keys(user_id: str, mode: str = "paper") -> Optional[tuple[str, str]]:
    """
    Retrieve and decrypt Alpaca API keys for a specific user and mode.

    This uses the alpaca_keys table with encrypted secrets.
    """
    try:
        response = (
            supabase.table("alpaca_keys")
            .select("*")
            .eq("user_id", user_id)
            .eq("mode", mode)
            .limit(1)
            .execute()
        )

        if not response.data:
            print(f"[{mode.upper()}] No Alpaca API keys found for user {user_id}")
            return None

        row = response.data[0]
        api_key_id = row["api_key_id"]
        api_secret_enc = row["api_secret_enc"]

        api_secret = fernet.decrypt(api_secret_enc.encode("utf-8")).decode("utf-8")

        return api_key_id, api_secret
    except Exception as e:
        print(f"[{mode.upper()}] Error retrieving or decrypting Alpaca keys for user {user_id}: {e}")
        return None


def upsert_bot_config(
    user_id: str,
    mode: str,
    is_running: bool,
    strategy_id: str = "default_rsi",
    last_error: Optional[str] = None,
) -> None:
    """
    Single source of truth for bot state, stored in bots_config.

    Worker processes watch this table and start/stop per-user bots accordingly.
    """
    payload: dict = {
        "user_id": user_id,
        "mode": mode,
        "strategy_id": strategy_id,
        "is_running": is_running,
        "updated_at": utc_now_iso(),
    }

    if last_error is not None:
        payload["last_error"] = last_error

    # Unique constraint (user_id, mode) ensures one bot per user per mode.
    supabase.table("bots_config").upsert(payload).execute()


def log_bot_event(user_id: str, mode: str, message: str) -> None:
    """
    Append a human-readable event into bot_logs so the UI can show a clean history.
    """
    try:
        supabase.table("bot_logs").insert(
            {
                "user_id": user_id,
                "mode": mode,
                "message": message,
                "created_at": utc_now_iso(),
            }
        ).execute()
    except Exception as e:
        # Never break user flow because of logging
        print(f"[{mode.upper()}] Warning: failed to write bot log for user {user_id}: {e}")


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "AIBOTIX Bot Server is up and running!"}


@app.get("/api/status")
@app.get("/status")
async def get_status_alias(user_id: Optional[str] = None):
    return await get_status(user_id)

async def get_status(user_id: Optional[str] = None):
    """
    Return bot running status.

    - If user_id is provided: return paper/live flags for that specific user.
    - If user_id is omitted: return global paper/live flags (any bot running).
    """
    try:
        if user_id:
            # Per-user status
            resp = (
                supabase.table("bots_config")
                .select("mode, is_running, updated_at, last_error")
                .eq("user_id", user_id)
                .execute()
            )

            paper_running = False
            live_running = False
            updated_at = utc_now_iso()
            last_error: Optional[str] = None

            for row in resp.data or []:
                mode = row.get("mode")
                if mode == "paper":
                    paper_running = bool(row.get("is_running"))
                elif mode == "live":
                    live_running = bool(row.get("is_running"))

                # Take the latest updated_at we see
                if row.get("updated_at"):
                    updated_at = row["updated_at"]

                if row.get("last_error"):
                    last_error = row["last_error"]

            return {
                "paper_running": paper_running,
                "live_running": live_running,
                "updated_at": updated_at,
                "last_error": last_error,
            }

        # Global status across all users (for admin/diagnostics)
        resp = supabase.table("bots_config").select("mode, is_running").execute()
        paper_running = any(
            row.get("mode") == "paper" and bool(row.get("is_running"))
            for row in (resp.data or [])
        )
        live_running = any(
            row.get("mode") == "live" and bool(row.get("is_running"))
            for row in (resp.data or [])
        )

        return {
            "paper_running": paper_running,
            "live_running": live_running,
            "updated_at": utc_now_iso(),
        }
    except Exception as e:
        print("Error in /api/status:", e)
        return {
            "paper_running": False,
            "live_running": False,
            "updated_at": utc_now_iso(),
            "error": "Failed to fetch status",
        }


@app.post("/api/start")
async def start_bot(request: Request):
    """
    Mark a bot as running for a specific user + mode.

    - Validates user_id and mode.
    - Ensures Alpaca keys exist (so the worker will be able to trade).
    - Optionally runs AI ticker selector for logging/preview.
    - Updates bots_config so worker will start or keep the bot running.
    """
    body = await request.json()
    user_id: Optional[str] = body.get("user_id")
    mode: str = body.get("mode", "paper")
    strategy_id: str = body.get("strategy_id", "default_rsi")

    if not user_id:
        return {"error": "Missing user_id in request."}

    if mode not in ("paper", "live"):
        return {"error": "Invalid mode. Expected 'paper' or 'live'."}

    # Ensure API keys exist for this user+mode before we mark bot as running
    creds = get_alpaca_keys(user_id, mode)
    if creds is None:
        msg = "No Alpaca API keys found for this user and mode."
        log_bot_event(user_id, mode, f"❌ {msg}")
        return {"error": msg}

    # Optional: run AI ticker selector for logging / preview.
    # Trading worker can still independently choose tickers.
    try:
        import bot.ai_ticker_selector_aibotix as ai_ticker_selector_aibotix

        results = await ai_ticker_selector_aibotix.stage_a_screen_and_collect(limit=5)
        ai_ticker_selector_aibotix.log_selected_tickers_for_learning(results)
        selected_tickers = ai_ticker_selector_aibotix.score_tickers(results)

        print(f"[{mode.upper()}] AI selected tickers for user {user_id}: {selected_tickers}")

        if selected_tickers:
            log_bot_event(
                user_id,
                mode,
                "AI selected tickers: " + ", ".join(selected_tickers),
            )
        else:
            log_bot_event(user_id, mode, "AI ticker selector returned no symbols.")
    except Exception as e:
        print(f"[{mode.upper()}] AI ticker selector failed for user {user_id}: {e}")
        log_bot_event(
            user_id,
            mode,
            "AI ticker selector error; trading will fall back to default logic.",
        )

    # Mark bot as running in bots_config; worker will observe and ensure a loop is alive.
    try:
        upsert_bot_config(
            user_id=user_id,
            mode=mode,
            is_running=True,
            strategy_id=strategy_id,
            last_error=None,
        )
        log_bot_event(user_id, mode, "✅ Bot start requested.")
    except Exception as e:
        print(f"[{mode.upper()}] Failed to update bots_config on start for user {user_id}: {e}")
        return {"error": "Failed to mark bot as running. Please try again."}

    return {
        "message": f"{mode.capitalize()} bot start requested for user.",
        "mode": mode,
        "user_id": user_id,
    }


@app.post("/api/stop")
async def stop_bot(request: Request):
    """
    Mark a bot as stopped for a specific user + mode.

    Worker will see is_running = false and gracefully wind down the loop.
    """
    body = await request.json()
    user_id: Optional[str] = body.get("user_id")
    mode: str = body.get("mode", "paper")

    if not user_id:
        return {"error": "Missing user_id in request."}

    if mode not in ("paper", "live"):
        return {"error": "Invalid mode. Expected 'paper' or 'live'."}

    try:
        upsert_bot_config(
            user_id=user_id,
            mode=mode,
            is_running=False,
        )
        log_bot_event(user_id, mode, "🛑 Bot stop requested.")
        return {
            "message": f"{mode.capitalize()} bot stop requested for user.",
            "mode": mode,
            "user_id": user_id,
        }
    except Exception as e:
        print(f"[{mode.upper()}] Failed to update bots_config on stop for user {user_id}: {e}")
        return {"error": "Failed to stop bot. Please try again."}


# --------------------------------------------------------------------
# Uvicorn entrypoint (Render calls `python main.py`)
# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting AIBOTIX bot server via uvicorn...")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        log_level="info",
    )