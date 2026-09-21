"""
Binance Demo Trading – automatische Orderausführung
=====================================================
Setzt bei einem Signal automatisch eine Markt-Einstiegsorder und eine
OCO-Order (One-Cancels-the-Other) für TP/SL auf Binance's offiziellem
Demo-Konto. Läuft mit echten, realistischen Binance-Marktdaten und
virtuellem Guthaben – kein finanzielles Risiko, aber auch keine 100%ige
Garantie auf exakt identische Kurse zur echten Börse (siehe Binance-Doku:
Demo-Kurse sind "ähnlich", nicht "identisch" zur Live-Börse).

Benötigt zwei GitHub-Secrets: BINANCE_DEMO_API_KEY, BINANCE_DEMO_API_SECRET
(erzeugt auf https://demo.binance.com/en/my/settings/api-management).
"""

import os
import time
import hmac
import hashlib
import logging
from decimal import Decimal, ROUND_HALF_UP
from urllib.parse import urlencode

import requests

logger = logging.getLogger("m15_signal_bot")

DEMO_API_KEY = os.getenv("BINANCE_DEMO_API_KEY")
DEMO_API_SECRET = os.getenv("BINANCE_DEMO_API_SECRET")
DEMO_BASE_URL = "https://demo-api.binance.com/api"

TRADE_SIZE_USDT = 100.0  # Positionsgröße pro Auto-Trade, in USDT (bzw. Quote-Währung)

_symbol_filters_cache = {}


def _signed_request(method: str, path: str, params: dict) -> dict:
    if not DEMO_API_KEY or not DEMO_API_SECRET:
        raise RuntimeError("BINANCE_DEMO_API_KEY / BINANCE_DEMO_API_SECRET fehlen (GitHub-Secrets prüfen).")

    params = dict(params)
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    query = urlencode(params)
    signature = hmac.new(DEMO_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    query += f"&signature={signature}"

    url = f"{DEMO_BASE_URL}{path}?{query}"
    headers = {"X-MBX-APIKEY": DEMO_API_KEY}

    resp = requests.request(method, url, headers=headers, timeout=15)
    if not resp.ok:
        # Binance liefert bei Fehlern eine sprechende JSON-Fehlermeldung – die
        # loggen wir mit, statt nur den nackten HTTP-Fehler zu zeigen.
        logger.error(f"Binance-Demo-API-Fehler ({resp.status_code}): {resp.text}")
    resp.raise_for_status()
    return resp.json()


def _get_symbol_filters(symbol: str) -> dict:
    """Holt Mindest-Schrittgrößen für Menge/Preis (Binance lehnt Orders ab,
    die nicht exakt auf diese Raster passen)."""
    if symbol in _symbol_filters_cache:
        return _symbol_filters_cache[symbol]

    resp = requests.get(f"{DEMO_BASE_URL}/v3/exchangeInfo", params={"symbol": symbol}, timeout=15)
    resp.raise_for_status()
    info = resp.json()["symbols"][0]

    filters = {"step_size": 0.0001, "tick_size": 0.01, "min_qty": 0.0, "min_notional": 0.0}
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            filters["step_size"] = float(f["stepSize"])
            filters["min_qty"] = float(f["minQty"])
        elif f["filterType"] == "PRICE_FILTER":
            filters["tick_size"] = float(f["tickSize"])
        elif f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
            filters["min_notional"] = float(f.get("minNotional", f.get("minNotionalValue", 0)))

    _symbol_filters_cache[symbol] = filters
    return filters


def _round_step(value: float, step: float) -> float:
    """Rundet einen Wert auf das von Binance vorgegebene Raster (z. B. Mengen
    dürfen nur in Schritten von 0.00001 BTC gehandelt werden). Nutzt Decimal,
    damit auch sehr kleine Schrittgrößen (z. B. 1e-05) korrekt behandelt werden."""
    if step <= 0:
        return value
    step_dec = Decimal(str(step))
    value_dec = Decimal(str(value))
    steps = (value_dec / step_dec).to_integral_value(rounding=ROUND_HALF_UP)
    result = steps * step_dec
    return float(result)


def place_demo_trade(symbol: str, side: str, entry_price: float, sl: float, tp: float) -> bool:
    """Platziert eine Markt-Einstiegsorder + eine OCO-Order (TP/SL) auf Binance
    Demo Trading. Gibt True zurück, wenn beide Orders erfolgreich platziert wurden."""
    try:
        filters = _get_symbol_filters(symbol)
        step = filters["step_size"]
        tick = filters["tick_size"]

        raw_qty = TRADE_SIZE_USDT / entry_price
        qty = _round_step(raw_qty, step)
        if qty < filters["min_qty"] or qty <= 0:
            logger.warning(f"Auto-Trade {symbol}: berechnete Menge {qty} unter Mindestmenge, übersprungen.")
            return False
        if qty * entry_price < filters["min_notional"]:
            logger.warning(f"Auto-Trade {symbol}: Orderwert unter Mindest-Notional, übersprungen.")
            return False

        entry_side = "BUY" if side == "BUY" else "SELL"
        exit_side = "SELL" if side == "BUY" else "BUY"

        # 1) Markt-Einstieg
        order = _signed_request("POST", "/v3/order", {
            "symbol": symbol, "side": entry_side, "type": "MARKET", "quantity": qty,
        })
        logger.info(f"Auto-Trade Einstieg platziert: {symbol} {entry_side} {qty} -> Order-ID {order.get('orderId')}")

        # 2) OCO-Order für TP/SL (automatischer Ausstieg, sobald einer der
        # beiden Werte erreicht wird; der jeweils andere wird dann storniert)
        tp_price = _round_step(tp, tick)
        sl_stop = _round_step(sl, tick)
        # Stop-Limit-Preis knapp hinter dem Stop-Trigger, damit die Order bei
        # schnellen Bewegungen realistisch noch ausgeführt wird.
        sl_limit = _round_step(sl * (0.999 if side == "BUY" else 1.001), tick)

        oco = _signed_request("POST", "/v3/order/oco", {
            "symbol": symbol, "side": exit_side, "quantity": qty,
            "price": tp_price,
            "stopPrice": sl_stop,
            "stopLimitPrice": sl_limit,
            "stopLimitTimeInForce": "GTC",
        })
        logger.info(f"Auto-Trade OCO (TP/SL) platziert: {symbol} {exit_side} -> Order-Liste {oco.get('orderListId')}")
        return True

    except requests.RequestException as e:
        logger.error(f"Auto-Trade fehlgeschlagen für {symbol}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unerwarteter Fehler beim Auto-Trade {symbol}: {e}")
        return False
