import ai_ticker_selector_aibotix
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
        "paper_running": bot_processes["paper"] is not None,
        "live_running": bot_processes["live"] is not None,
        "updated_at": datetime.utcnow().isoformat()
    }

@app.post("/api/start")
async def start_bot(request: Request):
    global bot_processes
    body = await request.json()
    user_id = body.get("user_id")
    mode = body.get("mode", "paper")

    if bot_processes[mode] is not None:
        return {"message": f"{mode.capitalize()} bot already running."}

    if not user_id:
        return {"error": "Missing user_id in request."}

    creds = get_alpaca_keys(user_id, mode)
    if creds is None:
        return {"error": "Could not retrieve API keys."}

    api_key_id, api_secret = creds
    selected_tickers = ai_ticker_selector_aibotix.get_top_tickers_from_api_keys(api_key_id, api_secret, mode)
    print(f"[{mode.upper()}] AI selected tickers: {selected_tickers}")

    if not selected_tickers:
        print(f"[{mode.upper()}] No tickers selected — waiting for market signals.")

    env = os.environ.copy()
    env["APCA_API_KEY_ID"] = api_key_id
    env["APCA_API_SECRET_KEY"] = api_secret

    bot_processes[mode] = subprocess.Popen(
        ["python", "aibotix_trading_bot.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
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
    global bot_processes
    body = await request.json()
    user_id = body.get("user_id")
    mode = body.get("mode", "paper")

    process = bot_processes.get(mode)
    if process:
        process.terminate()
        bot_processes[mode] = None
        print(f"[{mode.upper()}] Bot stopped successfully.")
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
    uvicorn.run(app, host="0.0.0.0", port=10000)