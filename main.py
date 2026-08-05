"""
M15/M5 Signal Bot – BTCUSDT
============================
Analysiert fortlaufend abgeschlossene Kerzen von Binance auf mehreren
Timeframes (EMA200, Bollinger Bänder 20/2.0, RSI14) und sendet bei
einem validen Setup eine Telegram-Push-Nachricht.

Es werden ZWEI Signal-Typen erkannt:

1. RE-TEST (Trendfortsetzung nach Rücksetzer):
   Kurs berührt ein Band, schließt aber wieder INNERHALB der Bänder
   zurück (Ablehnung/Bounce) – im Trend des EMA200.

2. BREAKOUT (Ausbruch mit Trendbestätigung):
   Kurs bricht durch ein Band und schließt AUSSERHALB davon (bleibt
   dort) – im Trend des EMA200, mit starkem RSI.

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

# Welche Symbole und Timeframes sollen geprüft werden?
SYMBOLS = ["BTCUSDT", "PAXGUSDT", "EURUSDT"]  # PAXGUSDT = Gold, EURUSDT = EUR/USD
TIMEFRAMES = ["15m", "5m"]

# Welche Signal-Kategorien aktiv geprüft/gesendet werden sollen.
# Per Backtest (60 Tage, 3.058 Trades) deaktiviert: BREAKOUT lief mit -17,19% klar
# negativ, RETEST dagegen mit +3,01% positiv. Breakout-Logik bleibt im Code
# erhalten, falls sie später (z. B. nach weiteren Anpassungen) reaktiviert wird.
ENABLED_CATEGORIES = ["RETEST"]  # Optionen: "RETEST", "BREAKOUT", "MOMENTUM"

# Mehrere Basis-URLs für die Kerzendaten: Binance blockiert seine Haupt-API teils
# nach Region (Fehler 451), z. B. wenn GitHub Actions zufällig einen US-Server zieht.
# data-api.binance.vision ist Binance's eigene, separate Domain nur für öffentliche
# Marktdaten und davon in der Regel nicht betroffen – wird deshalb zuerst versucht.
BINANCE_KLINES_URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
CANDLE_LIMIT = 300  # genug Historie für EMA200

EMA_PERIOD = 200
BB_LENGTH = 20
BB_STD = 2.0
RSI_PERIOD = 14

ATR_PERIOD = 14          # Anzahl Kerzen für die ATR-Berechnung (Standard: 14)
SL_RETEST_ATR_MULT = 1.5    # SL-Abstand beim Re-Test = 1,5x ATR
SL_BREAKOUT_ATR_MULT = 2.0  # SL-Abstand beim Breakout = 2x ATR (etwas mehr Puffer, da volatiler)
CRV = 1.5           # Chance-Risiko-Verhältnis für den Take Profit

RSI_RETEST_HIGH = 60   # Re-Test SELL: RSI muss mindestens so hoch sein
RSI_RETEST_LOW = 40    # Re-Test BUY: RSI muss höchstens so niedrig sein
RSI_BREAKOUT_HIGH = 60  # Breakout BUY: RSI muss mindestens so hoch sein
RSI_BREAKOUT_LOW = 40   # Breakout SELL: RSI muss höchstens so niedrig sein
BREAKOUT_LOOKBACK = 5   # Wie viele Kerzen nach dem eigentlichen Ausbruch die RSI-Bestätigung noch zählt

# --- Momentum-Strategie (aggressiver, EMA-Crossover statt Band-Berührung) ---
# Bewusst NOCH NICHT live (siehe ENABLED_CATEGORIES oben) – erst per Backtest
# bewerten, bevor überhaupt eine Nachricht verschickt wird.
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
RSI_MOMENTUM_HIGH = 55  # lockerer als Re-Test (60), da Momentum früher einsteigen will
RSI_MOMENTUM_LOW = 45
MOMENTUM_REQUIRE_PATTERN = False  # True = zusätzlich Kerzenmuster-Bestätigung nötig

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_signal.json")
OPEN_TRADES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_trades.json")
TRADE_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_history.json")
RETRY_SLEEP_SECONDS = 60  # Wartezeit nach einem Fehler, bevor erneut versucht wird

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m15_signal_bot")


# ---------------------------------------------------------------------------
# Daten & Indikatoren
# ---------------------------------------------------------------------------

def fetch_klines(symbol: str, interval: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """Holt die letzten Kerzen von der öffentlichen Binance-REST-API.
    Probiert dabei mehrere Basis-URLs durch, falls eine davon (z. B. wegen
    regionaler Sperrung, Fehler 451) nicht erreichbar ist."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_error = None

    for base_url in BINANCE_KLINES_URLS:
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as e:
            last_error = e
            logger.warning(f"Kline-Abruf über {base_url} fehlgeschlagen ({e}), versuche nächste URL.")
    else:
        raise last_error

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
    """Berechnet EMA200, Bollinger Bänder (20, 2.0), RSI(14) und ATR(14)."""
    df = df.copy()

    df["ema200"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()

    sma = df["close"].rolling(BB_LENGTH).mean()
    std = df["close"].rolling(BB_LENGTH).std(ddof=0)
    df["bb_mid"] = sma
    df["bb_upper"] = sma + BB_STD * std
    df["bb_lower"] = sma - BB_STD * std

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # ATR (Average True Range) – misst die durchschnittliche Kerzen-Schwankungsbreite
    # der letzten ATR_PERIOD Kerzen. Skaliert SL/TP automatisch an die tatsächliche
    # Volatilität jedes Assets/Timeframes, statt eines starren Prozentsatzes.
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(ATR_PERIOD).mean()

    return df


# ---------------------------------------------------------------------------
# Hilfsfunktion für SL/TP
# ---------------------------------------------------------------------------

def build_signal(signal_type: str, category: str, entry: float, sl: float, rsi: float, candle_time) -> dict:
    if signal_type == "SELL":
        risk = sl - entry
        tp = entry - risk * CRV
    else:
        risk = entry - sl
        tp = entry + risk * CRV
    return {
        "type": signal_type,
        "category": category,  # "RETEST" oder "BREAKOUT"
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rsi": rsi,
        "candle_time": candle_time,
    }


# ---------------------------------------------------------------------------
# Signal-Logik: RE-TEST (Rücksetzer/Ablehnung am Band, zurück in den Bändern)
# ---------------------------------------------------------------------------

def check_retest_signal(df: pd.DataFrame):
    if len(df) < EMA_PERIOD + 5:
        return None

    candle = df.iloc[-2]
    close, open_, high, low = candle["close"], candle["open"], candle["high"], candle["low"]
    ema200, bb_upper, bb_lower, rsi, atr = (
        candle["ema200"], candle["bb_upper"], candle["bb_lower"], candle["rsi"], candle["atr"]
    )

    if any(pd.isna(x) for x in [ema200, bb_upper, bb_lower, rsi, atr]):
        return None

    is_red = close < open_
    is_green = close > open_

    # SELL: Abwärtstrend, Hoch berührt oberes Band, schließt wieder darunter (Ablehnung)
    if close < ema200 and high >= bb_upper and close < bb_upper and is_red and rsi >= RSI_RETEST_HIGH:
        sl = close + atr * SL_RETEST_ATR_MULT
        return build_signal("SELL", "RETEST", close, sl, rsi, candle["close_time"])

    # BUY: Aufwärtstrend, Tief berührt unteres Band, schließt wieder darüber (Bounce)
    if close > ema200 and low <= bb_lower and close > bb_lower and is_green and rsi <= RSI_RETEST_LOW:
        sl = close - atr * SL_RETEST_ATR_MULT
        return build_signal("BUY", "RETEST", close, sl, rsi, candle["close_time"])

    return None


# ---------------------------------------------------------------------------
# Signal-Logik: BREAKOUT (Ausbruch mit Trendbestätigung, bleibt außerhalb)
# ---------------------------------------------------------------------------

def _find_breakout_start(df: pd.DataFrame, current_pos: int, band_col: str, is_outside) -> int:
    """Läuft von der aktuellen Kerze rückwärts, solange der Kurs durchgehend außerhalb
    des Bandes liegt, und gibt die Position der ERSTEN Kerze dieses Ausbruchs zurück.
    Bricht spätestens nach BREAKOUT_LOOKBACK+1 Schritten ab – das genügt, weil ein
    länger zurückliegender Ausbruch ohnehin nicht mehr als "aktuell" zählt (siehe
    BREAKOUT_LOOKBACK-Prüfung beim Aufruf). Das hält die Funktion auch bei sehr
    langer Kerzen-Historie (z. B. im Backtest) schnell."""
    pos = current_pos
    min_pos = max(0, current_pos - BREAKOUT_LOOKBACK - 1)
    while pos > min_pos and is_outside(df.iloc[pos - 1]["close"], df.iloc[pos - 1][band_col]):
        pos -= 1
    return pos


def check_breakout_signal(df: pd.DataFrame):
    if len(df) < EMA_PERIOD + BREAKOUT_LOOKBACK + 2:
        return None

    current_pos = len(df) - 2  # Position von iloc[-2]
    candle = df.iloc[current_pos]

    close, open_ = candle["close"], candle["open"]
    ema200, bb_upper, bb_lower, rsi, atr = (
        candle["ema200"], candle["bb_upper"], candle["bb_lower"], candle["rsi"], candle["atr"]
    )

    if any(pd.isna(x) for x in [ema200, bb_upper, bb_lower, rsi, atr]):
        return None

    # --- BUY-Breakout: Kurs liegt (seit kurzem) über dem oberen Band, im Aufwärtstrend ---
    if close > ema200 and close > bb_upper:
        breakout_start = _find_breakout_start(df, current_pos, "bb_upper", lambda c, b: c > b)
        duration = current_pos - breakout_start  # 0 = genau die Ausbruchskerze selbst

        if duration <= BREAKOUT_LOOKBACK:
            # RSI darf seit Ausbruchsbeginn noch NICHT die Schwelle erreicht haben –
            # sonst wäre das Signal schon einmal gesendet worden (verhindert Dauerfeuer).
            # Die Farbe der einzelnen Kerzen spielt hier bewusst keine Rolle mehr: der RSI
            # kann seine Schwelle auch auf einer kurzen "Verschnaufpause"-Kerze überschreiten.
            rsi_already_confirmed = any(
                df.iloc[k]["rsi"] >= RSI_BREAKOUT_HIGH for k in range(breakout_start, current_pos)
            )
            if not rsi_already_confirmed and rsi >= RSI_BREAKOUT_HIGH:
                sl = close - atr * SL_BREAKOUT_ATR_MULT
                return build_signal("BUY", "BREAKOUT", close, sl, rsi, candle["close_time"])

    # --- SELL-Breakout: Kurs liegt (seit kurzem) unter dem unteren Band, im Abwärtstrend ---
    if close < ema200 and close < bb_lower:
        breakout_start = _find_breakout_start(df, current_pos, "bb_lower", lambda c, b: c < b)
        duration = current_pos - breakout_start

        if duration <= BREAKOUT_LOOKBACK:
            rsi_already_confirmed = any(
                df.iloc[k]["rsi"] <= RSI_BREAKOUT_LOW for k in range(breakout_start, current_pos)
            )
            if not rsi_already_confirmed and rsi <= RSI_BREAKOUT_LOW:
                sl = close + atr * SL_BREAKOUT_ATR_MULT
                return build_signal("SELL", "BREAKOUT", close, sl, rsi, candle["close_time"])

    return None


# ---------------------------------------------------------------------------
# Kerzenmuster-Erkennung (für die optionale Momentum-Bestätigung)
# ---------------------------------------------------------------------------

def _is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    prev_red = prev["close"] < prev["open"]
    curr_green = curr["close"] > curr["open"]
    engulfs = curr["open"] <= prev["close"] and curr["close"] >= prev["open"]
    return prev_red and curr_green and engulfs


def _is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    prev_green = prev["close"] > prev["open"]
    curr_red = curr["close"] < curr["open"]
    engulfs = curr["open"] >= prev["close"] and curr["close"] <= prev["open"]
    return prev_green and curr_red and engulfs


def _is_hammer(candle: pd.Series) -> bool:
    body = abs(candle["close"] - candle["open"])
    if body == 0:
        return False
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    return lower_wick >= body * 2 and upper_wick <= body * 0.5


def _is_shooting_star(candle: pd.Series) -> bool:
    body = abs(candle["close"] - candle["open"])
    if body == 0:
        return False
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    return upper_wick >= body * 2 and lower_wick <= body * 0.5


# ---------------------------------------------------------------------------
# Signal-Logik: MOMENTUM (EMA-Crossover, aggressiver als Re-Test)
# ---------------------------------------------------------------------------

def check_momentum_signal(df: pd.DataFrame, require_pattern: bool = MOMENTUM_REQUIRE_PATTERN):
    if len(df) < EMA_PERIOD + 5:
        return None

    current_pos = len(df) - 2
    candle = df.iloc[current_pos]
    prev = df.iloc[current_pos - 1]

    close, ema200, ema_fast, ema_slow, rsi, atr = (
        candle["close"], candle["ema200"], candle["ema_fast"], candle["ema_slow"], candle["rsi"], candle["atr"]
    )
    prev_fast, prev_slow = prev["ema_fast"], prev["ema_slow"]

    if any(pd.isna(x) for x in [ema200, ema_fast, ema_slow, rsi, atr, prev_fast, prev_slow]):
        return None

    bullish_cross = prev_fast <= prev_slow and ema_fast > ema_slow
    bearish_cross = prev_fast >= prev_slow and ema_fast < ema_slow

    # BUY: EMA9 kreuzt EMA21 von unten nach oben, im Aufwärtstrend, RSI zeigt Stärke
    if bullish_cross and close > ema200 and rsi >= RSI_MOMENTUM_HIGH:
        if require_pattern and not (_is_bullish_engulfing(prev, candle) or _is_hammer(candle)):
            return None
        sl = close - atr * SL_RETEST_ATR_MULT
        return build_signal("BUY", "MOMENTUM", close, sl, rsi, candle["close_time"])

    # SELL: EMA9 kreuzt EMA21 von oben nach unten, im Abwärtstrend, RSI zeigt Schwäche
    if bearish_cross and close < ema200 and rsi <= RSI_MOMENTUM_LOW:
        if require_pattern and not (_is_bearish_engulfing(prev, candle) or _is_shooting_star(candle)):
            return None
        sl = close + atr * SL_RETEST_ATR_MULT
        return build_signal("SELL", "MOMENTUM", close, sl, rsi, candle["close_time"])

    return None


# ---------------------------------------------------------------------------
# Duplikat-Schutz (State-Datei)
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f)


def load_state() -> dict:
    return _load_json(STATE_FILE)


def save_state(state: dict):
    _save_json(STATE_FILE, state)


def load_open_trades() -> dict:
    return _load_json(OPEN_TRADES_FILE)


def save_open_trades(trades: dict):
    _save_json(OPEN_TRADES_FILE, trades)


def append_trade_history(entry: dict):
    history = _load_json(TRADE_HISTORY_FILE)
    entries = history.get("entries", [])
    entries.append(entry)
    _save_json(TRADE_HISTORY_FILE, {"entries": entries})


# ---------------------------------------------------------------------------
# Offene Trades verfolgen (TP/SL-Ergebnis)
# ---------------------------------------------------------------------------

def check_open_trades_for_symbol(symbol: str, interval: str, df: pd.DataFrame, open_trades: dict) -> bool:
    """Prüft alle offenen Trades dieses Symbol/Timeframe gegen die Kerzen SEIT
    Trade-Eröffnung. Sendet bei TP/SL-Treffer eine Ergebnis-Nachricht und
    entfernt den Trade aus den offenen Positionen. Gibt True zurück bei Änderung."""
    changed = False
    key_prefix = f"{symbol}_{interval}_"
    relevant_ids = [tid for tid in open_trades if tid.startswith(key_prefix)]

    for trade_id in relevant_ids:
        trade = open_trades[trade_id]
        opened_at = pd.Timestamp(trade["candle_time"])

        # Nur Kerzen NACH der Signal-Kerze zählen (die Signal-Kerze selbst war der Einstieg).
        subsequent = df[df["close_time"] > opened_at]
        if subsequent.empty:
            continue

        outcome = None
        exit_price = None

        for _, candle in subsequent.iterrows():
            if trade["type"] == "BUY":
                hit_sl = candle["low"] <= trade["sl"]
                hit_tp = candle["high"] >= trade["tp"]
            else:  # SELL
                hit_sl = candle["high"] >= trade["sl"]
                hit_tp = candle["low"] <= trade["tp"]

            if hit_sl and hit_tp:
                # Beides in derselben Kerze getroffen – nicht eindeutig, welches zuerst
                # geschah. Wir gehen konservativ vom schlechteren Fall (SL) aus.
                outcome, exit_price = "SL", trade["sl"]
                break
            elif hit_sl:
                outcome, exit_price = "SL", trade["sl"]
                break
            elif hit_tp:
                outcome, exit_price = "TP", trade["tp"]
                break

        if outcome is None:
            continue  # Trade noch offen, weder TP noch SL erreicht

        message = format_outcome_message(trade, outcome, exit_price)
        if send_telegram_message(message):
            logger.info(f"Trade abgeschlossen: {trade_id} -> {outcome} @ {exit_price}")
            append_trade_history({**trade, "outcome": outcome, "exit_price": exit_price})
            del open_trades[trade_id]
            changed = True
        else:
            logger.warning(f"Trade-Ergebnis erkannt ({trade_id} -> {outcome}), aber Telegram-Versand fehlgeschlagen.")

    return changed


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def _decimals_for_price(price: float) -> int:
    """Wählt eine sinnvolle Anzahl Nachkommastellen je nach Preisniveau –
    bei EUR/USD (~1$) braucht es mehr Nachkommastellen als bei BTC (~60.000$),
    sonst sehen SL und TP in der Nachricht identisch aus."""
    if price < 10:
        return 5
    if price < 1000:
        return 3
    return 2


CATEGORY_LABELS = {"RETEST": "Re-Test", "BREAKOUT": "Breakout", "MOMENTUM": "Momentum"}
CATEGORY_REASONS = {
    "RETEST": "Abprall/Ablehnung am Bollinger Band im Trend des EMA 200",
    "BREAKOUT": "Ausbruch durchs Bollinger Band mit Trendbestätigung (EMA 200 + starker RSI)",
    "MOMENTUM": "EMA9/EMA21-Crossover im Trend des EMA 200 mit RSI-Bestätigung",
}


def format_signal_message(symbol: str, interval: str, signal: dict) -> str:
    category_label = CATEGORY_LABELS.get(signal["category"], signal["category"])
    reason = CATEGORY_REASONS.get(signal["category"], "")

    display_symbol = symbol.replace("USDT", "USD")
    d = _decimals_for_price(signal["entry"])
    return (
        f"🚨 *{interval.upper()} SIGNAL {signal['type']} ({category_label}) - {display_symbol}* 🚨\n\n"
        f"- *Einstieg:* {signal['entry']:.{d}f} $\n"
        f"- *Stop Loss (SL):* {signal['sl']:.{d}f} $\n"
        f"- *Take Profit (TP):* {signal['tp']:.{d}f} $\n"
        f"- *CRV:* 1:{CRV}\n"
        f"- *RSI:* {signal['rsi']:.2f}\n"
        f"- *Grund:* {reason}"
    )


def format_outcome_message(trade: dict, outcome: str, exit_price: float) -> str:
    display_symbol = trade["symbol"].replace("USDT", "USD")
    category_label = CATEGORY_LABELS.get(trade["category"], trade["category"])
    d = _decimals_for_price(trade["entry"])

    if outcome == "TP":
        emoji = "✅"
        title = "TAKE PROFIT ERREICHT"
        pnl = (exit_price - trade["entry"]) if trade["type"] == "BUY" else (trade["entry"] - exit_price)
    else:
        emoji = "❌"
        title = "STOP LOSS ERREICHT"
        pnl = (exit_price - trade["entry"]) if trade["type"] == "BUY" else (trade["entry"] - exit_price)

    return (
        f"{emoji} *{title} - {trade['interval'].upper()} {trade['type']} ({category_label}) - {display_symbol}* {emoji}\n\n"
        f"- *Einstieg:* {trade['entry']:.{d}f} $\n"
        f"- *Ausstieg:* {exit_price:.{d}f} $\n"
        f"- *Ergebnis:* {'+' if pnl >= 0 else ''}{pnl:.{d}f} $"
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

def process_symbol_timeframe(symbol: str, interval: str, state: dict, open_trades: dict) -> bool:
    """Prüft ein Symbol/Timeframe auf beide Signal-Typen UND auf TP/SL-Ergebnisse
    offener Trades. Gibt True zurück, wenn state oder open_trades geändert wurden."""
    changed = False
    try:
        df = fetch_klines(symbol, interval)
        df = calculate_indicators(df)
    except requests.RequestException as e:
        logger.error(f"Netzwerkfehler beim Abruf von {symbol} {interval}: {e}")
        return False

    # Zuerst offene Trades gegen die aktuellen Kerzen prüfen (TP/SL-Ergebnis).
    if check_open_trades_for_symbol(symbol, interval, df, open_trades):
        changed = True

    possible_signals = []
    if "RETEST" in ENABLED_CATEGORIES:
        possible_signals.append(check_retest_signal(df))
    if "BREAKOUT" in ENABLED_CATEGORIES:
        possible_signals.append(check_breakout_signal(df))
    if "MOMENTUM" in ENABLED_CATEGORIES:
        possible_signals.append(check_momentum_signal(df))
    signals = [s for s in possible_signals if s is not None]

    # Detailliertes Debug-Log der letzten ABGESCHLOSSENEN Kerze (iloc[-2]) –
    # damit sich spätere "hätte das nicht ein Signal geben müssen?"-Fälle im
    # Actions-Log direkt anhand der tatsächlichen Zahlen nachvollziehen lassen.
    last = df.iloc[-2]
    logger.info(
        f"{symbol} {interval} | close={last['close']:.5f} ema200={last['ema200']:.5f} "
        f"bb_upper={last['bb_upper']:.5f} bb_lower={last['bb_lower']:.5f} rsi={last['rsi']:.2f} "
        f"atr={last['atr']:.5f} | Kerzenzeit={last['close_time']}"
    )

    if not signals:
        logger.info(f"{symbol} {interval}: keine Signal-Bedingung erfüllt.")
        return changed

    for signal in signals:
        state_key = f"{symbol}_{interval}_{signal['category']}"
        candle_time_str = str(signal["candle_time"])

        if state.get(state_key) == candle_time_str:
            logger.info(f"{symbol} {interval} {signal['category']}: bereits gesendet – überspringe.")
            continue

        message = format_signal_message(symbol, interval, signal)
        if send_telegram_message(message):
            state[state_key] = candle_time_str
            changed = True
            logger.info(f"Signal gesendet: {symbol} {interval} {signal['category']} {signal['type']} @ {signal['entry']:.2f}")

            # Als offenen Trade speichern, damit TP/SL künftig verfolgt werden.
            trade_id = f"{symbol}_{interval}_{signal['category']}_{candle_time_str}"
            open_trades[trade_id] = {
                "symbol": symbol,
                "interval": interval,
                "category": signal["category"],
                "type": signal["type"],
                "entry": signal["entry"],
                "sl": signal["sl"],
                "tp": signal["tp"],
                "rsi": signal["rsi"],
                "candle_time": candle_time_str,
            }
        else:
            logger.warning(f"Signal erkannt ({symbol} {interval} {signal['category']}), aber Telegram-Versand fehlgeschlagen.")

    return changed


def run_check():
    state = load_state()
    open_trades = load_open_trades()
    state_changed = False
    trades_changed = False

    for symbol in SYMBOLS:
        for interval in TIMEFRAMES:
            if process_symbol_timeframe(symbol, interval, state, open_trades):
                state_changed = True
                trades_changed = True

    if state_changed:
        save_state(state)
    if trades_changed:
        save_open_trades(open_trades)


def seconds_until_next_check() -> float:
    """Berechnet die Wartezeit bis kurz nach dem Schluss der nächsten 5-Minuten-Kerze
    (deckt damit automatisch auch jeden 15-Minuten-Kerzenschluss mit ab)."""
    now = datetime.now(timezone.utc)
    next_slot_minute = ((now.minute // 5) + 1) * 5

    if next_slot_minute >= 60:
        next_dt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_dt = now.replace(minute=next_slot_minute, second=0, microsecond=0)

    next_dt += timedelta(seconds=10)
    return max((next_dt - now).total_seconds(), 5)


def main():
    logger.info(f"Signal Bot gestartet für {SYMBOLS} auf {TIMEFRAMES}.")
    while True:
        try:
            run_check()
            sleep_seconds = seconds_until_next_check()
        except requests.RequestException as e:
            logger.error(f"Netzwerkfehler: {e} – neuer Versuch in {RETRY_SLEEP_SECONDS}s.")
            sleep_seconds = RETRY_SLEEP_SECONDS
        except Exception as e:
            logger.exception(f"Unerwarteter Fehler: {e} – neuer Versuch in {RETRY_SLEEP_SECONDS}s.")
            sleep_seconds = RETRY_SLEEP_SECONDS

        logger.info(f"Nächste Prüfung in {int(sleep_seconds)} Sekunden.")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
