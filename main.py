import ai_ticker_selector_aibotix
from fastapi import FastAPI, Request
from threading import Thread
import subprocess
from supabase import create_client, Client
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ENCRYPTION_KEY = os.getenv("SECRET_ENCRYPTION_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
fernet = Fernet(ENCRYPTION_KEY.encode())

app = FastAPI()

bot_process = None

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
def get_status():
    return {"running": bot_process is not None}

@app.post("/api/start")
async def start_bot(request: Request):
    global bot_process
    if bot_process is not None:
        return {"message": "Bot already running."}

    body = await request.json()
    user_id = body.get("user_id")
    mode = body.get("mode", "paper")

    if not user_id:
        return {"error": "Missing user_id in request."}

    creds = get_alpaca_keys(user_id, mode)
    if creds is None:
        return {"error": "Could not retrieve API keys."}

    api_key_id, api_secret = creds
    selected_tickers = ai_ticker_selector_aibotix.get_top_tickers_from_api_keys(api_key_id, api_secret, mode)
    print("Selected tickers:", selected_tickers)

    if not selected_tickers:
        print("No tickers selected. Bot will wait for market signals.")

    env = os.environ.copy()
    env["APCA_API_KEY_ID"] = api_key_id
    env["APCA_API_SECRET_KEY"] = api_secret

    bot_process = subprocess.Popen(
        ["python", "aibotix_trading_bot.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    return {"message": "Bot started for user."}

@app.post("/api/stop")
def stop_bot():
    global bot_process
    if bot_process is None:
        return {"message": "Bot not running."}
    bot_process.terminate()
    bot_process = None
    return {"message": "Bot stopped."}