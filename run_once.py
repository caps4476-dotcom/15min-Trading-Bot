"""
Einmalige Ausführung des Signal-Checks – MIT Testnachricht.
"""

import logging
from main import run_check, send_telegram_message

logger = logging.getLogger("m15_signal_bot")

if __name__ == "__main__":
    try:
        send_telegram_message("✅ Testnachricht – die Verbindung funktioniert!")
        run_check()
    except Exception as e:
        logger.exception(f"Fehler beim Signal-Check: {e}")
