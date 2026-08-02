"""
Einmalige Ausführung des Signal-Checks.

Für den Dauerbetrieb auf einem eigenen Server nutzt du `main.py` (Endlosschleife).
Für GitHub Actions (kostenloser Betrieb ohne eigenen Server) nutzt du dieses
Skript: GitHub Actions ruft es per Zeitplan (Cron) alle 15 Minuten auf, jeder
Aufruf prüft die letzte abgeschlossene Kerze genau einmal und beendet sich
danach wieder.
"""

import logging
from main import run_check

logger = logging.getLogger("m15_signal_bot")

if __name__ == "__main__":
    try:
        run_check()
    except Exception as e:
        # Darf den GitHub-Actions-Lauf nicht als "failed" markieren,
        # sonst gibt es bei jedem Netzwerkhänger eine Fehler-Mail von GitHub.
        logger.exception(f"Fehler beim Signal-Check: {e}")
