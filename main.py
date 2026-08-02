"""
M15 Signal Bot – BTCUSDT
=========================
Analysiert fortlaufend abgeschlossene 15-Minuten-Kerzen von Binance
(EMA200, Bollinger Bänder 20/2.0, RSI14) und sendet bei einem validen
Setup (Trend + Re-Test am Bollinger Band) eine Telegram-Push-Nachricht.

Es wird bewusst NUR die letzte ABGESCHLOSSENE Kerze (iloc[-2]) bewertet,
da die aktuelle Kerze (iloc[-1]) noch läuft und sich verändern kann.
"""

import os
import time
import json
import logging
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
CANDLE_LIMIT = 300  # genug Historie für EMA200

EMA_PERIOD = 200
BB_LENGTH = 20
BB_STD = 2.0
RSI_PERIOD = 14

SL_BUFFER = 15.0    # $ Puffer über/unter der Signal-Kerze für den Stop Loss
CRV = 1.5           # Chance-Risiko-Verhältnis für den Take Profit

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_signal.json")
RETRY_SLEEP_SECONDS = 60  # Wartezeit nach einem Fehler, bevor erneut versucht wird

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m15_signal_bot")


# ---------------------------------------------------------------------------
# Daten & Indikatoren
# ---------------------------------------------------------------------------

def fetch_klines(symbol: str = SYMBOL, interval: str = INTERVAL, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """Holt die letzten Kerzen von der öffentlichen Binance-REST-API."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet EMA200, Bollinger Bänder (20, 2.0) und RSI(14)."""
    df = df.copy()

    # EMA 200
    df["ema200"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    # Bollinger Bänder
    sma = df["close"].rolling(BB_LENGTH).mean()
    std = df["close"].rolling(BB_LENGTH).std(ddof=0)
    df["bb_mid"] = sma
    df["bb_upper"] = sma + BB_STD * std
    df["bb_lower"] = sma - BB_STD * std

    # RSI (Wilder-Glättung via EWM)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    return df


# ---------------------------------------------------------------------------
# Signal-Logik
# ---------------------------------------------------------------------------

def check_signal(df: pd.DataFrame):
    """Prüft ausschließlich die letzte ABGESCHLOSSENE Kerze (iloc[-2])."""
    if len(df) < EMA_PERIOD + 5:
        return None

    candle = df.iloc[-2]

    close = candle["close"]
    open_ = candle["open"]
    high = candle["high"]
    low = candle["low"]
    ema200 = candle["ema200"]
    bb_upper = candle["bb_upper"]
    bb_lower = candle["bb_lower"]
    rsi = candle["rsi"]

    if any(pd.isna(x) for x in [ema200, bb_upper, bb_lower, rsi]):
        return None

    is_red = close < open_
    is_green = close > open_

    # --- SELL: Abwärtstrend & Re-Test am oberen Band ---
    if (
        close < ema200
        and high >= bb_upper
        and close < bb_upper
        and is_red
        and rsi >= 60
    ):
        sl = high + SL_BUFFER
        risk = sl - close
        tp = close - risk * CRV
        return {
            "type": "SELL",
            "entry": close,
            "sl": sl,
            "tp": tp,
            "rsi": rsi,
            "candle_time": candle["close_time"],
        }

    # --- BUY: Aufwärtstrend & Re-Test am unteren Band ---
    if (
        close > ema200
        and low <= bb_lower
        and close > bb_lower
        and is_green
        and rsi <= 40
    ):
        sl = low - SL_BUFFER
        risk = close - sl
        tp = close + risk * CRV
        return {
            "type": "BUY",
            "entry": close,
            "sl": sl,
            "tp": tp,
            "rsi": rsi,
            "candle_time": candle["close_time"],
        }

    return None


# ---------------------------------------------------------------------------
# Duplikat-Schutz (State-Datei)
# ---------------------------------------------------------------------------

def load_last_signal_time():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        return data.get("last_candle_time")
    except (json.JSONDecodeError, OSError):
        return None


def save_last_signal_time(candle_time_str: str):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_candle_time": candle_time_str}, f)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def format_signal_message(signal: dict) -> str:
    reason = "Abprall am Bollinger Band im Trend des EMA 200"
    return (
        f"🚨 *M15 SIGNAL {signal['type']} - BTCUSD* 🚨\n\n"
        f"- *Einstieg:* {signal['entry']:.2f} $\n"
        f"- *Stop Loss (SL):* {signal['sl']:.2f} $\n"
        f"- *Take Profit (TP):* {signal['tp']:.2f} $\n"
        f"- *CRV:* 1:{CRV}\n"
        f"- *RSI:* {signal['rsi']:.2f}\n"
        f"- *Grund:* {reason}"
    )


def send_telegram_message(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_TOKEN oder TELEGRAM_CHAT_ID fehlt in der .env-Datei.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.error(f"Telegram-Nachricht konnte nicht gesendet werden: {e}")
        return False


# ---------------------------------------------------------------------------
# Ablaufsteuerung
# ---------------------------------------------------------------------------

def run_check():
    df = fetch_klines()
    df = calculate_indicators(df)
    signal = check_signal(df)

    if signal is None:
        logger.info("Keine Signal-Bedingung erfüllt.")
        return

    candle_time_str = str(signal["candle_time"])
    last_sent = load_last_signal_time()

    if last_sent == candle_time_str:
        logger.info("Signal für diese Kerze wurde bereits gesendet – überspringe (Duplikat-Schutz).")
        return

    message = format_signal_message(signal)
    if send_telegram_message(message):
        save_last_signal_time(candle_time_str)
        logger.info(f"Signal gesendet: {signal['type']} @ {signal['entry']:.2f}")
    else:
        logger.warning("Signal erkannt, aber Telegram-Versand ist fehlgeschlagen (State nicht gespeichert).")


def seconds_until_next_check() -> float:
    """Berechnet die Wartezeit bis kurz nach dem Schluss der nächsten 15-Minuten-Kerze."""
    now = datetime.now(timezone.utc)
    next_slot_minute = ((now.minute // 15) + 1) * 15

    if next_slot_minute >= 60:
        next_dt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_dt = now.replace(minute=next_slot_minute, second=0, microsecond=0)

    # Sicherheitspuffer, damit die Binance-Kerze garantiert schon geschlossen ist
    next_dt += timedelta(seconds=20)
    return max((next_dt - now).total_seconds(), 5)


def main():
    logger.info(f"M15 Signal Bot gestartet für {SYMBOL} ({INTERVAL}).")
    while True:
        try:
            run_check()
            sleep_seconds = seconds_until_next_check()
        except requests.RequestException as e:
            logger.error(f"Netzwerkfehler: {e} – neuer Versuch in {RETRY_SLEEP_SECONDS}s.")
            sleep_seconds = RETRY_SLEEP_SECONDS
        except Exception as e:
            # Fängt alles Unerwartete ab, damit der Bot niemals abstürzt.
            logger.exception(f"Unerwarteter Fehler: {e} – neuer Versuch in {RETRY_SLEEP_SECONDS}s.")
            sleep_seconds = RETRY_SLEEP_SECONDS

        logger.info(f"Nächste Prüfung in {int(sleep_seconds)} Sekunden.")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
