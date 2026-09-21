"""
Verbindungstest für Binance Demo Trading.
Fragt NUR den Kontostand ab (kein Order-Versand) – dient allein dazu, zu
prüfen, ob die API-Schlüssel korrekt sind und die Signierung funktioniert,
bevor irgendeine echte (wenn auch virtuelle) Order riskiert wird.
"""

import logging
from binance_demo_trader import _signed_request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("connection_test")


def main():
    logger.info("Verbindungstest zu Binance Demo Trading gestartet ...")
    try:
        account = _signed_request("GET", "/v3/account", {})
    except Exception as e:
        logger.error(f"Verbindung fehlgeschlagen: {e}")
        raise SystemExit(1)

    logger.info("Verbindung erfolgreich! Konto-Guthaben (nur Werte > 0):")
    balances = [
        b for b in account.get("balances", [])
        if float(b["free"]) > 0 or float(b["locked"]) > 0
    ]
    if not balances:
        logger.info("(Keine Guthaben > 0 gefunden – Konto evtl. noch leer.)")
    for b in balances:
        logger.info(f"  {b['asset']}: frei={b['free']} gesperrt={b['locked']}")

    can_trade = account.get("canTrade")
    logger.info(f"Trading-Berechtigung auf diesem API-Key aktiv: {can_trade}")


if __name__ == "__main__":
    main()
