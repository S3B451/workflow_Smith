import os
import sqlite3
import json
import requests
from datetime import datetime

class WorkflowInterface:
    def __init__(self, db_path="workflow_results.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialisiert eine lokale SQLite Datenbank, falls sie nicht existiert."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS runs 
                           (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                            timestamp TEXT, 
                            metrics TEXT, 
                            results TEXT)''')

    def export_to_markdown(self, data: dict, filename="performance_tagebuch.md"):
        """Speichert die Daten als Markdown."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md_content = f"\n# 📅 Run: {timestamp}\n"
        
        # Tabelle für Metriken
        md_content += "| Modell | Speed (t/s) | VRAM (MB) |\n| :--- | :---: | :---: |\n"
        for m in data.get("metrics", []):
            md_content += f"| {m['model']} | {m['speed_tps']} | {m['vram_mb']} |\n"
        
        with open(filename, "a", encoding="utf-8") as f:
            f.write(md_content + "\n---\n")
        print(f"📁 Markdown: {filename} aktualisiert.")

    def export_to_sql(self, data: dict):
        """Speichert den gesamten State in einer SQL-Datenbank."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics_json = json.dumps(data.get("metrics", []))
        # Wir filtern nur die Ergebnisse heraus
        results_only = {k: v for k, v in data.items() if "out" in k or "Instruct" in k}
        results_json = json.dumps(results_only)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO runs (timestamp, metrics, results) VALUES (?, ?, ?)",
                         (timestamp, metrics_json, results_json))
        print(f"🗄️ SQL: Daten in '{self.db_path}' gespeichert.")

    def export_to_api(self, data: dict, url="http://localhost:8000/ingest"):
        """Versendet die Daten an eine REST-API (Mockup)."""
        try:
            # response = requests.post(url, json=data, timeout=5)
            print(f"🌐 API: Daten-Payload bereit für {url} (aktuell gemockt).")
        except Exception as e:
            print(f"❌ API-Fehler: {e}")

    def run_all_exports(self, data: dict):
        """Führt alle aktiven Schnittstellen gleichzeitig aus."""
        self.export_to_markdown(data)
        self.export_to_sql(data)
        self.export_to_api(data)

# In 02_simple.py
from workflow_interface import WorkflowInterface

# # ... manager, logger initialisieren ...
# interface = WorkflowInterface()
# nodes = SpecializedNodes(manager, logger, interface) # Interface mitgeben!

# In SpecializedNodes Klasse (nodes.py)

    # def reporter_node(self, state: dict):
    #     print("\n--- Reporter: Übergebe Daten an Interface-Modul ---")
        
    #     # Der Node delegiert die Arbeit komplett nach außen
    #     self.interface.run_all_exports(state)
        
    #     return {"report_complete": True}