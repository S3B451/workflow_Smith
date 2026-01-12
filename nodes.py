
"""ÜBERSICHT DER REGISTERIERTEN NODES
---------------------------------------------------------------------------
MODELL-NAME (Key)                          | NODE-NAME (Funktion)
---------------------------------------------------------------------------
meta-llama/Llama-3.2-3B-Instruct           | nodes.llama_3_2_3_b_node
mistralai/Ministral-3-3B-Instruct-2512     | nodes.ministral_3_node
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    | nodes.deepseek_r1_7b_node
Mistral-7B-Instruct-v0.3                   | nodes.mistral_7b_node
google/gemma-3n-E2B-it                     | nodes.gemma_2b_node
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  | nodes.deepseek_r1_1_5b_node
rd211/Qwen3-1.7B-Instruct                  | nodes.qwen_3_1_7b_node

(Finalisierung & Export)                   | nodes.reporter_node
---------------------------------------------------------------------------"""


from typing import Dict, Any
import torch
from PIL import Image
import time
import functools
from datetime import datetime
from console_feedback import ActivitySpinner

def get_ts(): 
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log_node_performance(model_key: str): # model_key ist z.B. "meta-llama/..."
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
            print(f"[{get_ts()}] [NODE] START: {func.__name__}")
            start_time = time.time()
            
            # HIER DIE ÄNDERUNG: Nur 2 Werte entpacken
            response, token_count = func(self, state)
            
            duration = time.time() - start_time
            # Wir nutzen model_key direkt für den Logger
            metrics = self.logger.log_step(model_key, duration, token_count)
            
            new_state = state.copy()
            new_state[model_key] = response
            new_state["metrics"] = state.get("metrics", []) + [metrics]
            
            return new_state
        return wrapper
    return decorator


class SpecializedNodes:
    def __init__(self, manager, logger, interface):
        """
        Initialisiert die Nodes mit dem ModelManager und dem WorkflowLogger.
        """
        self.spinner = ActivitySpinner()
        self.manager = manager
        self.logger = logger
        self.interface = interface

    # 1. Llama 3.2 Node (Analyst)
    @log_node_performance("meta-llama/Llama-3.2-3B-Instruct")
    def llama_3_2_3_b_node(self, state: Dict[str, Any]):
        """Node für die Portfolio-Analyse mit Llama 3.2."""
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        prompt = f"Analysiere folgendes Portfolio auf Risiko: {state.get('portfolio_items')}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.spinner.start("Generiere Antwort...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        self.spinner.stop()

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Rückgabe für den Decorator: (Antwort-Text, Token-Anzahl)
        return response, outputs[0].shape[0]

    @log_node_performance("Qwen/Qwen2.5-7B-Instruct")
    def qwen_2_5_7B_node(self, state: Dict[str, Any]):
        """Hochpräzise Analyse-Node mit Qwen 2.5 7B."""
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        portfolio = state.get('portfolio_items', 'Keine Daten')
        
        # Qwen ist extrem gut darin, Zahlen und Fakten zu korrelieren
        prompt = f"""Analysiere die folgenden Portfolio-Positionen. 
        Erstelle eine kurze Tabelle mit:
        1. Risiko-Score (1-10)
        2. Diversifikations-Beitrag
        3. Empfehlung (Halten/Umschichten)
        
        Daten: {portfolio}"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.spinner.start("Qwen 2.5 berechnet Präzisions-Check...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.3)
        self.spinner.stop()

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, outputs[0].shape[0]
    
    # 3. DeepSeek R1 Node (Teacher/Optimizer)
    @log_node_performance("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    def deepseek_r1_7b_node(self, state: Dict[str, Any]):
        """Optimierungs-Node mit DeepSeek R1."""
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        portfolio = state.get('portfolio_items', 'Keine Daten')
        # Greift auf das Ergebnis des Analysten (Llama) zu
        analyst_feedback = state.get('meta-llama/Llama-3.2-3B-Instruct', 'Kein Feedback')
        
        prompt = f"Portfolio: {portfolio}\nAnalyst: {analyst_feedback}\nVerbesserungsvorschläge:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        self.spinner.start("Generiere Antwort...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.6
            )
        self.spinner.stop()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, outputs[0].shape[0]

    # 4. Mistral 7B Node (Strategist/Finalist)
    @log_node_performance("Mistral-7B-Instruct-v0.3")
    def mistral_7b_node(self, state: Dict[str, Any]):
        """Node für das finale Fazit mit Mistral 7B."""
        model_name = "Mistral-7B-Instruct-v0.3"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        portfolio = state.get('portfolio_items', 'Keine Daten')
        analyst_out = state.get('meta-llama/Llama-3.2-3B-Instruct', 'Keine Analyse')
        teacher_out = state.get('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'Keine Zweitmeinung')
        
        prompt = f"Erstelle Fazit für {portfolio} aus {analyst_out} und {teacher_out}:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        self.spinner.start("Generiere Antwort...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400, 
                do_sample=True, 
                temperature=0.7
            )
        self.spinner.stop()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, outputs[0].shape[0]

    # 5. Google Gemma Node
    @log_node_performance("google/gemma-3n-E2B-it")
    def gemma_2b_node(self, state: Dict[str, Any]):
        """Node für die Verarbeitung mit Google Gemma 2B."""
        model_name = "google/gemma-3n-E2B-it"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        # Beispielhafter Prompt für dieses Modell
        prompt = f"Fasse die Kernrisiken für folgendes Portfolio zusammen: {state.get('portfolio_items')}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.spinner.start("Gemma analysiert...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=250)
        self.spinner.stop()

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Rückgabe für den Decorator: (Antwort-Text, Token-Anzahl)
        return response, outputs[0].shape[0]

    # 6. DeepSeek R1 1.5B Node
    @log_node_performance("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    def deepseek_r1_1_5b_node(self, state: Dict[str, Any]):
        """Node für Reasoning und Logik-Check mit DeepSeek R1 1.5B."""
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        # DeepSeek R1 nutzt oft <reasoning> Tags oder Chain-of-Thought
        prompt = f"Prüfe die bisherigen Ergebnisse auf logische Konsistenz: {state.get('portfolio_items')}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.spinner.start("DeepSeek denkt nach (Reasoning)...")
        with torch.no_grad():
            # Höheres Token-Limit, da Reasoning-Modelle oft längere interne Denkprozesse haben
            outputs = model.generate(**inputs, max_new_tokens=450, temperature=0.6)
        self.spinner.stop()

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Rückgabe für den Decorator: (Antwort-Text, Token-Anzahl)
        return response, outputs[0].shape[0]
    # 7. Qwen 1.7B Node
    @log_node_performance("rd211/Qwen3-1.7B-Instruct")
    def qwen_3_1_7b_node(self, state: Dict[str, Any]):
        """Node für extrem schnelle Kurz-Analysen mit Qwen 1.7B."""
        model_name = "rd211/Qwen3-1.7B-Instruct"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        prompt = f"Erstelle eine ultrakurze Liste der 3 wichtigsten Assets: {state.get('portfolio_items')}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.spinner.start("Qwen 1.7B arbeitet...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150)
        self.spinner.stop()

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, outputs[0].shape[0]

    # 8. Reporter Node (VRAM-sicher & Dynamisch)
    def reporter_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n---  Reporter: Generiere finalen Bericht ---")
        
        #  DEBUG: Zeige uns, was wirklich im State gelandet ist
        print(f"DEBUG: Vorhandene Keys im State: {list(state.keys())}") 

        internal_keys = ['portfolio_items', 'metrics', 'report', 'report_complete', 'final_data']
        report_sections = []
        report_sections.append("===  DYNAMISCHER PORTFOLIO BERICHT ===")
        report_sections.append(f"Zeitstempel: {datetime.now().strftime('%H:%M:%S')}\n")

        found_content = False
        for key, value in state.items():
            # Wir ignorieren interne Felder
            if key in internal_keys:
                continue
            
            # Falls der Decorator ein Tupel (Text, Tokens) speichert, extrahieren wir nur den Text
            content = ""
            if isinstance(value, tuple) and len(value) > 0:
                content = str(value[0])
            elif isinstance(value, str):
                content = value
            
            if content and len(content) > 10: # Nur Sektionen mit echtem Inhalt
                found_content = True
                clean_name = key.split('/')[-1] if '/' in key else key
                report_sections.append(f"🔹 ABSCHNITT: {clean_name.upper()}")
                report_sections.append(f"{content[:1000]}") # Erste 1000 Zeichen
                report_sections.append("-" * 40)

        if not found_content:
            report_sections.append(" HINWEIS: Keine Analyse-Inhalte in den Nodes gefunden.")
            report_sections.append(f"Gefundene Keys: {list(state.keys())}")

        full_report = "\n".join(report_sections)

        #  Interface-Speicherung
        if hasattr(self, 'interface') and self.interface:
            self.interface.save_all(state)

        return {
            "report": full_report,
            "report_complete": True
        }
        



#macht nur aerger wegen tokenizer
    # # 2. Ministral Node
    # @log_node_performance("mistralai/Ministral-3-3B-Instruct-2512")
    # def ministral_3_node(self, state: Dict[str, Any]):
    #     """Node für die Portfolio-Analyse mit Ministral."""
    #     model_name = "mistralai/Ministral-3-3B-Instruct-2512"
    #     model, tokenizer = self.manager.load_by_name(model_name)
        
    #     prompt = f"Analysiere folgendes Portfolio auf Risiko: {state.get('portfolio_items')}"
    #     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    #     self.spinner.start("Generiere Antwort...")
    #     with torch.no_grad():
    #         outputs = model.generate(**inputs, max_new_tokens=200)
    #     self.spinner.stop()

    #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return response, outputs[0].shape[0]

    # macht nur aerger wegen transformators version
    # @log_node_performance("microsoft/Phi-4-mini-instruct")
    # def phi_4_mini_node(self, state: Dict[str, Any]):
    #     """Hochpräzise Analyse-Node mit Microsoft Phi-4-mini."""
    #     model_name = "microsoft/Phi-4-mini-instruct"
    #     model, tokenizer = self.manager.load_by_name(model_name)
        
    #     # Phi-4 ist exzellent darin, Datenpunkte exakt zu bewerten
    #     prompt = f"Bewertet die folgenden Portfolio-Positionen auf einer Skala von 1-10 bezüglich Risiko und Rendite: {state.get('portfolio_items')}"
        
    #     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
    #     self.spinner.start("Phi-4 analysiert präzise...")
    #     with torch.no_grad():
    #         outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.4)
    #     self.spinner.stop()

    #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return response, outputs[0].shape[0]   
# def florence_2_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node für Florenz 2 (Vision-Modell - aktuell deaktiviert)."""
#     print("--- Node: Florenz 2 wird aktiv ---")
#     
#     model, processor = self.manager.load_by_name("microsoft/Florence-2-large", is_vision=True)
#     
#     prompt = f"Optimiere folgendes Portfolio: {state.get('portfolio_items')}"
#     
#     # Dummy-Bild für Vision-Modell
#     dummy_image = Image.new('RGB', (100, 100), color='black')
#     
#     inputs = processor(text=prompt, images=dummy_image, return_tensors="pt").to("cuda")
#     
#     # DTYPE-FIX für float16
#     inputs = {
#         k: v.to(torch.float16) if v.dtype == torch.float else v 
#         for k, v in inputs.items()
#     }
#     
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=inputs["input_ids"],
#             pixel_values=inputs["pixel_values"],
#             max_new_tokens=200,
#             do_sample=False,
#             num_beams=1,
#             use_cache=False
#         )
#     
#     response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#     return {"microsoft/Florence-2-large": response}