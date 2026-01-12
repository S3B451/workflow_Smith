import os 
import sqlite3
import json
from datetime import datetime

class WorkflowInterface:
    def __init__(self):
        # Bestimmt den absoluten Pfad des Ordners, in dem dieses Skript liegt
        self.base_dir = os.path.dirname(os.path.abspath(__file__)) 
        # Definiert den Zielordner für alle Ausgaben
        # 1. Output: Ordner für Markdown Berichte
        self.md_output_dir = os.path.join(self.base_dir, "md_output") 
        # 2. Output: Ordner für die Datenbank (dAbA
        self.dAbA_dir = os.path.join(self.base_dir, "dAbA") 
        # Erstellt den Ordner 'output', falls er nicht existiert
        os.makedirs(self.md_output_dir, exist_ok=True) 
        os.makedirs(self.dAbA_dir, exist_ok=True) 
        # Pfade für DB und Markdown innerhalb des Output-Ordners
        self.db_path = os.path.join(self.dAbA_dir, "performance.db") 
        self.md_path = os.path.join(self.md_output_dir, "tagebuch.md") 
        
        self._init_db()

    def _init_db(self):
        """Initialisiert die SQL-Datenbank am neuen Pfad."""
        with sqlite3.connect(self.db_path) as conn: 
            conn.execute('''CREATE TABLE IF NOT EXISTS history 
                           (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                            timestamp TEXT, 
                            metrics TEXT,
                            summary TEXT)''')

    def save_all(self, state: dict):
        from workflow_logger import get_ts 
        timestamp = get_ts() 
        
        metrics = state.get("metrics", [])
        
        self._export_markdown(timestamp, metrics)
        self._export_sql(timestamp, metrics)
        
        print(f"[{timestamp}] [INTF] Berichte -> {self.md_output_dir}") 
        print(f"[{timestamp}] [INTF] Datenbank -> {self.dAbA_dir}") 

    def _export_markdown(self, timestamp, metrics):
        #  Nutzt den sicheren Pfad im Output-Ordner
        with open(self.md_path, "a", encoding="utf-8") as f: 
            f.write(f"\n##  Run am {timestamp}\n")
            f.write("| Modell | Zeit | Speed | VRAM |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for m in metrics:
                vram_gb = m['vram_mb'] / 1024
                f.write(f"| {m['model']} | {m['duration_sec']}s | {m['speed_tps']} t/s | {vram_gb:.2f} GB |\n")

    def _export_sql(self, timestamp, metrics):
        metrics_json = json.dumps(metrics)
        with sqlite3.connect(self.db_path) as conn: 
            conn.execute("INSERT INTO history (timestamp, metrics, summary) VALUES (?, ?, ?)", 
                         (timestamp, metrics_json, "Workflow erfolgreich beendet"))