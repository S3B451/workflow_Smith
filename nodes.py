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

    # 2. Ministral Node
    @log_node_performance("mistralai/Ministral-3-3B-Instruct-2512")
    def ministral_3_node(self, state: Dict[str, Any]):
        """Node für die Portfolio-Analyse mit Ministral."""
        model_name = "mistralai/Ministral-3-3B-Instruct-2512"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        prompt = f"Analysiere folgendes Portfolio auf Risiko: {state.get('portfolio_items')}"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        self.spinner.start("Generiere Antwort...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        self.spinner.stop()

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, outputs[0].shape[0]

    # 3. DeepSeek R1 Node (Teacher/Optimizer)
    @log_node_performance("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    def deepseek_r1_7b_node(self, state: Dict[str, Any]):
        """Optimierungs-Node mit DeepSeek R1."""
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        model, tokenizer = self.manager.load_by_name(model_name, is_vision=False)
        
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
        model, tokenizer = self.manager.load_by_name(model_name, is_vision=False)
        
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

    # 5. Reporter Node (Zusammenfassung & Metriken)
    def reporter_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n--- Reporter: Finalisiere Bericht ---")

        # 1. Texte sicher aus dem State holen (Keys prüfen!)
        # WICHTIG: Die Keys müssen EXAKT so heißen wie im @log_node_performance Decorator
        llama_text = state.get('meta-llama/Llama-3.2-3B-Instruct', 'Keine Analyse vorhanden')
        deep_text = state.get('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'Kein Feedback vorhanden')
        mistral_text = state.get('Mistral-7B-Instruct-v0.3', 'Kein Fazit vorhanden')

        # 2. Den Bericht-String zusammenbauen
        full_report = (
            f"=== PORTFOLIO BERICHT ===\n\n"
            f"ANALYSRE (Llama):\n{llama_text[:500]}\n\n"
            f"OPTIMIERUNG (DeepSeek):\n{deep_text[:500]}\n\n"
            f"STRATEGIE (Mistral):\n{mistral_text[:500]}\n"
        )

        # 3. Das Interface nutzen (Schnittstelle nach außen)
        if hasattr(self, 'interface'):
            self.interface.save_all(state)

        # 4. WICHTIG: Den Key "report" zurückgeben, damit 02_simple.py ihn findet!
        return {
            "report": full_report,
            "report_complete": True
        }
        
    
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