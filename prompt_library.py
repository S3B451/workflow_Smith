import os

class PortfolioPrompts:
    """Zentrale Bibliothek für die Analyse von Mietregulierungen."""

    # --- KONSTANTEN ---
    BASE_DIR = "c:/Users/volks/Documents/pythonWorkflows/workflow_Smith"
    DEFAULT_PORTFOLIO = os.path.join(BASE_DIR, "01_Quelle.txt")

    @staticmethod
    def _read_source(source: str) -> str:
        """Liest die Quelle als reinen Text ein."""
        if not source:
            return "Keine Daten übergeben."
        # Falls ein Pfad übergeben wurde, diesen einlesen
        if isinstance(source, str) and os.path.exists(source):
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"[Fehler beim Lesen der Datei: {e}]"
        return str(source)

    # --- CONFIGS FÜR DIE NODES (Synchronisiert mit nodes.py) ---

    @classmethod
    def get_analyst_config(cls, source: str):
        content = cls._read_source(source)
        return {
            "system": "Du bist ein Experte für Wohnungspolitik und Mietrecht.",
            "user": f"Aufgabe 1: Analysiere die Formen der Mietregulierung. Erkläre den Unterschied "
                    f"zwischen 'strict price ceilings', 'vacancy control' und 'vacancy decontrol'.\n\n"
                    f"Quelle: {content}",
            "params": {"temperature": 0.3, "max_new_tokens": 500, "do_sample": True}
        }

    @classmethod
    def get_qwen_config(cls, source: str):
        content = cls._read_source(source)
        return {
            "system": "Du bist ein präziser Daten-Extraktor.",
            "user": f"Aufgabe 2: Erstelle eine Liste aller Länder/Städte und deren spezifische "
                    f"Prozentsätze für Mieterhöhungen (z.B. China 5%, Frankreich 3.5%).\n\n"
                    f"Quelle: {content}",
            "params": {"temperature": 0.05, "max_new_tokens": 400, "do_sample": True}
        }

    @classmethod
    def get_deepseek_config(cls, text: str):
        return {
            "system": "Du bist ein ökonomischer Auditor.",
            "user": f"Bewerte die ökonomischen Auswirkungen von Mietkontrollen und den Konsens "
                    f"unter Ökonomen basierend auf diesen Daten:\n{text}",
            "params": {"temperature": 0.6, "max_new_tokens": 600, "do_sample": True}
        }

    @classmethod
    def get_mistral_config(cls, audit_insight: str):
        return {
            "system": "Du bist ein Senior-Politikberater.",
            "user": f"Entwirf eine Empfehlung für eine ausgewogene Wohnungspolitik unter Nutzung "
                    f"der im Text genannten Alternativen.\n\nAudit: {audit_insight}",
            "params": {"temperature": 0.4, "max_new_tokens": 500, "do_sample": True}
        }