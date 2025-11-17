import bot.ai_ticker_selector_aibotix as ai_ticker_selector_aibotix
from bot.aibotix_trading_bot import init_trading_client, trade_loop_async, request_bot_stop
import asyncio
from fastapi import FastAPI, Request
from threading import Thread
import subprocess
from supabase import create_client, Client
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

try:
    print("🚀 Starting AIBOTIX bot server...")

    print("Initializing secure environment variables... ✔️")

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY or not ENCRYPTION_KEY:
        raise EnvironmentError("❌ Missing one or more required environment variables.")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    fernet = Fernet(ENCRYPTION_KEY.encode())
    print("✅ Supabase client and Fernet key initialized successfully.")

except Exception as e:
    import traceback
    print("❌ Fatal error during startup:", e)
    traceback.print_exc()
    raise e  # ensure the app exits

app = FastAPI()

active_bots = {"paper": {}, "live": {}}
bot_processes = {"paper": None, "live": None}

def get_alpaca_keys(user_id: str, mode: str = "paper"):
    try:
        response = (
            supabase
            .table("alpaca_keys")
            .select("*")
            .eq("user_id", user_id)
            .eq("mode", mode)
            .limit(1)
            .execute()
        )

        if not response.data:
            print(f"No API keys found for user {user_id} in mode {mode}")
            return None

        key_data = response.data[0]
        api_key_id = key_data["api_key_id"]
        api_secret_enc = key_data["api_secret_enc"]

        api_secret = fernet.decrypt(api_secret_enc.encode()).decode()

        return api_key_id, api_secret
    except Exception as e:
        print("Error retrieving or decrypting API keys:", e)
        return None

@app.get("/")
def read_root():
    return {"message": "AIBOTIX Bot Server is up and running!"}

@app.get("/api/status")
async def get_status():
    return {
        "paper_running": any(active_bots["paper"].values()),
        "live_running": any(active_bots["live"].values()),
        "updated_at": datetime.utcnow().isoformat()
    }

@app.post("/api/start")
async def start_bot(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    mode = body.get("mode", "paper")

    if any(active_bots[mode].values()):
        return {"message": f"{mode.capitalize()} bot already running."}

    if not user_id:
        return {"error": "Missing user_id in request."}

    creds = get_alpaca_keys(user_id, mode)
    if creds is None:
        return {"error": "Could not retrieve API keys."}

    api_key_id, api_secret = creds

    # Run Stage A: Screen and collect data (async)
    results = await ai_ticker_selector_aibotix.stage_a_screen_and_collect(limit=5)

    # Log results for learning
    ai_ticker_selector_aibotix.log_selected_tickers_for_learning(results)

    # Score and rank tickers
    selected_tickers = ai_ticker_selector_aibotix.score_tickers(results)

    print(f"[{mode.upper()}] AI selected tickers: {selected_tickers}")

    if not selected_tickers:
        print(f"[{mode.upper()}] No tickers selected — waiting for market signals.")

    # Initialize dynamic per-user trading client
    init_trading_client(api_key_id, api_secret, paper=(mode == "paper"))

    # Launch user-specific async trading task
    task = asyncio.create_task(trade_loop_async())
    active_bots[mode][user_id] = task
    try:
        supabase.table("bot_status").upsert({
            "user_id": user_id,
            "mode": mode,
            "running": True,
            "start_time": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        print(f"[{mode.upper()}] Warning: could not update bot status on start.")
    return {"message": "Bot started for user."}

@app.post("/api/stop")
async def stop_bot(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    mode = body.get("mode", "paper")

    task = active_bots[mode].get(user_id)
    if task:
        # Ask the trading bot loop to stop cleanly
        request_bot_stop()

        # Cancel the asyncio task if still alive
        try:
            task.cancel()
        except Exception:
            pass

        active_bots[mode][user_id] = None
        print(f"[{mode.upper()}] Bot stop requested successfully.")

        try:
            supabase.table("bot_status").upsert({
                "user_id": user_id,
                "mode": mode,
                "running": False,
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            print(f"[{mode.upper()}] Warning: could not update bot status on stop.")

        return {"message": f"{mode.capitalize()} bot stopped successfully."}
    else:
        return {"message": f"No {mode} bot is currently running."}

if __name__ == "__main__":
    import uvicorn
    import os

    print("🚀 Starting AIBOTIX bot server via uvicorn...")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        log_level="info"
    )