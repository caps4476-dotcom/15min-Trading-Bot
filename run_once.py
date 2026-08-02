"""
Einmalige Ausführung des Signal-Checks für GitHub Actions.
"""

import logging
from main import run_check

logger = logging.getLogger("m15_signal_bot")

if __name__ == "__main__":
    try:
        run_check()
    except Exception as e:
        logger.exception(f"Fehler beim Signal-Check: {e}")
