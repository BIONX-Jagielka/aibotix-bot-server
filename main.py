import ai_ticker_selector_aibotix
from fastapi import FastAPI
from threading import Thread
import subprocess

app = FastAPI()

bot_process = None

@app.get("/")
def read_root():
    return {"message": "AIBOTIX Bot Server is up and running!"}

@app.get("/api/status")
def get_status():
    return {"running": bot_process is not None}

@app.post("/api/start")
def start_bot():
    global bot_process
    if bot_process is not None:
        return {"message": "Bot already running."}
    selected_tickers = ai_ticker_selector_aibotix.get_top_tickers()  # Example function
    print("Selected tickers:", selected_tickers)

    if not selected_tickers:
        print("No tickers selected. Bot will wait for market signals.")

    bot_process = subprocess.Popen(
        ["python", "aibotix_trading_bot.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return {"message": "Bot started."}

@app.post("/api/stop")
def stop_bot():
    global bot_process
    if bot_process is None:
        return {"message": "Bot not running."}
    bot_process.terminate()
    bot_process = None
    return {"message": "Bot stopped."}

@app.get("/api/tickers")
def get_selected_tickers():
    return {"tickers": ai_ticker_selector_aibotix.get_top_tickers()}