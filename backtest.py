"""
Backtest – prüft die exakt gleichen Signal-Regeln aus main.py gegen historische
Kursdaten, um ehrliche Statistiken zu bekommen (Trefferquote, Durchschnitts-
gewinn/-verlust) statt anhand einzelner Chart-Screenshots zu raten.

Nutzt bewusst DIESELBEN Funktionen aus main.py (Indikator-Berechnung, Signal-
Erkennung), damit die Ergebnisse garantiert zur Live-Logik passen und nicht aus
Versehen auseinanderlaufen.
"""

import time
import logging
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

from main import (
    SYMBOLS, TIMEFRAMES, BINANCE_KLINES_URLS,
    EMA_PERIOD, BREAKOUT_LOOKBACK, CRV,
    calculate_indicators, check_retest_signal, check_breakout_signal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")

BACKTEST_DAYS = 60  # Zeitraum der historischen Daten
REPORT_FILE = "backtest_report.md"


# ---------------------------------------------------------------------------
# Historische Daten laden (mit Pagination, da Binance max. 1000 Kerzen/Anfrage liefert)
# ---------------------------------------------------------------------------

def fetch_historical_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000

    all_rows = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": current_start, "limit": 1000,
        }
        data = None
        for base_url in BINANCE_KLINES_URLS:
            try:
                resp = requests.get(base_url, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as e:
                logger.warning(f"Abruf über {base_url} fehlgeschlagen ({e}), versuche nächste URL.")
        if not data:
            break

        all_rows.extend(data)
        last_open_time = data[-1][0]
        if len(data) < 1000 or last_open_time <= current_start:
            break
        current_start = last_open_time + 1
        time.sleep(0.3)  # Binance-Rate-Limit schonen

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Simulation: Signale über die gesamte Historie erkennen + TP/SL-Ergebnis prüfen
# ---------------------------------------------------------------------------

def simulate(df: pd.DataFrame) -> list:
    """Läuft Kerze für Kerze durch die Historie (wie der Live-Bot es täte) und
    sammelt jedes gefundene Signal inkl. späterem TP/SL-Ergebnis."""
    trades = []
    start_i = EMA_PERIOD + BREAKOUT_LOOKBACK + 5

    for i in range(start_i, len(df) - 1):
        window = df.iloc[:i + 2]  # so dass window.iloc[-2] == df.iloc[i]

        for check_fn, category in [(check_retest_signal, "RETEST"), (check_breakout_signal, "BREAKOUT")]:
            signal = check_fn(window)
            if signal is None:
                continue

            entry, sl, tp, sig_type = signal["entry"], signal["sl"], signal["tp"], signal["type"]

            # Ergebnis in den folgenden Kerzen suchen
            outcome, exit_price = None, None
            for j in range(i + 1, len(df)):
                candle = df.iloc[j]
                if sig_type == "BUY":
                    hit_sl, hit_tp = candle["low"] <= sl, candle["high"] >= tp
                else:
                    hit_sl, hit_tp = candle["high"] >= sl, candle["low"] <= tp

                if hit_sl:
                    outcome, exit_price = "SL", sl
                    break
                elif hit_tp:
                    outcome, exit_price = "TP", tp
                    break

            pnl_pct = None
            if outcome:
                pnl_pct = ((exit_price - entry) / entry * 100) if sig_type == "BUY" else ((entry - exit_price) / entry * 100)
            trades.append({
                "category": category, "type": sig_type,
                "entry": entry, "sl": sl, "tp": tp,
                "outcome": outcome or "OFFEN",  # kein TP/SL bis Ende der Historie erreicht
                "pnl_pct": pnl_pct,
                "candle_time": signal["candle_time"],
            })

    return trades


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------

def summarize(trades: list) -> dict:
    closed = [t for t in trades if t["outcome"] in ("TP", "SL")]
    wins = [t for t in closed if t["outcome"] == "TP"]
    losses = [t for t in closed if t["outcome"] == "SL"]
    open_trades = [t for t in trades if t["outcome"] == "OFFEN"]

    win_rate = (len(wins) / len(closed) * 100) if closed else 0.0
    avg_win_pct = (sum(t["pnl_pct"] for t in wins) / len(wins)) if wins else 0.0
    avg_loss_pct = (sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 0.0
    total_pnl_pct = sum(t["pnl_pct"] for t in closed) if closed else 0.0

    return {
        "total_signals": len(trades),
        "closed": len(closed),
        "open": len(open_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "total_pnl_pct": total_pnl_pct,
    }


def main():
    logger.info(f"Backtest gestartet – {BACKTEST_DAYS} Tage, Symbole: {SYMBOLS}, Timeframes: {TIMEFRAMES}")

    report_lines = [
        f"# Backtest-Report",
        f"",
        f"Zeitraum: letzte {BACKTEST_DAYS} Tage | Erstellt: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"",
        f"| Symbol | Timeframe | Kategorie | Signale | Geschlossen | Offen | Wins | Losses | Trefferquote | Ø Gewinn | Ø Verlust | Gesamt-PnL |",
        f"|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    grand_total_pnl = 0.0
    grand_total_closed = 0
    grand_total_wins = 0

    for symbol in SYMBOLS:
        for interval in TIMEFRAMES:
            logger.info(f"Lade Historie für {symbol} {interval} ...")
            df = fetch_historical_klines(symbol, interval, BACKTEST_DAYS)
            if df.empty or len(df) < EMA_PERIOD + BREAKOUT_LOOKBACK + 10:
                logger.warning(f"Zu wenig Daten für {symbol} {interval}, überspringe.")
                continue

            df = calculate_indicators(df)
            trades = simulate(df)
            logger.info(f"{symbol} {interval}: {len(trades)} Signale gefunden in {len(df)} Kerzen.")

            for category in ("RETEST", "BREAKOUT"):
                cat_trades = [t for t in trades if t["category"] == category]
                if not cat_trades:
                    continue
                s = summarize(cat_trades)
                grand_total_pnl += s["total_pnl_pct"]
                grand_total_closed += s["closed"]
                grand_total_wins += s["wins"]

                report_lines.append(
                    f"| {symbol} | {interval} | {category} | {s['total_signals']} | {s['closed']} | "
                    f"{s['open']} | {s['wins']} | {s['losses']} | {s['win_rate']:.1f}% | "
                    f"{s['avg_win_pct']:+.2f}% | {s['avg_loss_pct']:+.2f}% | {s['total_pnl_pct']:+.2f}% |"
                )

    overall_win_rate = (grand_total_wins / grand_total_closed * 100) if grand_total_closed else 0.0
    report_lines += [
        f"",
        f"## Gesamt",
        f"",
        f"- Geschlossene Trades insgesamt: {grand_total_closed}",
        f"- Gesamt-Trefferquote: {overall_win_rate:.1f}%",
        f"- Summe aller PnL-Prozentsätze: {grand_total_pnl:+.2f}%",
        f"",
        f"*Hinweis: PnL wird pro Trade in % vom Einstiegskurs berechnet und einfach aufsummiert "
        f"(keine Zinseszins-/Kapitalgrößen-Berücksichtigung). Das ist eine Annäherung, um die "
        f"Regeln über viele Trades hinweg zu vergleichen – kein Ersatz für echtes Money-Management.*",
    ]

    report = "\n".join(report_lines)
    with open(REPORT_FILE, "w") as f:
        f.write(report)

    logger.info("Backtest abgeschlossen. Report:\n" + report)


if __name__ == "__main__":
    main()
