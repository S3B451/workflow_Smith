from typing import Dict, Any
import torch
import time
import functools
from datetime import datetime
from console_feedback import ActivitySpinner
from prompt_library import PortfolioPrompts

def get_ts(): 
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log_node_performance(model_key: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
            print(f"[{get_ts()}] [NODE] START: {func.__name__}")
            start_time = time.time()
            
            # Inferenz ausführen
            response, token_count = func(self, state)
            
            # --- NEU: VORSCHAU DER ERSTEN 10 ZEILEN ---
            lines = response.splitlines()
            preview = "\n".join(lines[:10])
            print(f"\n---  VORSCHAU ({model_key}) ---")
            print(preview)
            if len(lines) > 10:
                print(f"... [+ {len(lines) - 10} weitere Zeilen]")
            print("-" * 40 + "\n")
            # ------------------------------------------

            duration = time.time() - start_time
            metrics = self.logger.log_step(model_key, duration, token_count)
            
            new_state = state.copy()
            new_state[model_key] = response
            new_state["metrics"] = state.get("metrics", []) + [metrics]
            
            return new_state
        return wrapper
    return decorator

class SpecializedNodes:
    def __init__(self, manager, logger, interface):
        self.manager = manager
        self.logger = logger
        self.interface = interface
        self.spinner = ActivitySpinner()

    
    @log_node_performance("llama_out")
    def llama_1b_test_node(self, state: Dict[str, Any]):
        print(f"\n{'='*20} 🔍 DEBUG START {'='*20}")
        
        # 1. Was kommt aus dem STATE?
        source_path = state.get("portfolio_items")
        print(f"[STEP 1] Rohdaten aus State (portfolio_items): {source_path}")

        # 2. Was macht die LIBRARY daraus?
        # Hier wird die Methode in prompt_library.py aufgerufen
        config = PortfolioPrompts.get_analyst_config(source_path)
        print(f"[STEP 2] Config von Library erhalten:")
        print(f"   -> System-Anweisung: {config['system']}")
        print(f"   -> User-Aufgabe (gekürzt): {config['user'][:100]}...")
        print(f"   -> KI-Parameter: {config['params']}")

        # 3. Der finale PROMPT (Das Llama-Template)
        # Hier fügen wir die Einzelteile in das Format ein, das Llama versteht
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{config['system']}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{config['user']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        print(f"[STEP 3] Finaler Prompt-String bereit für GPU (Länge: {len(prompt)} Zeichen)")

        # 4. Laden und Inferenz
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        print(f"[STEP 4] Inferenz läuft auf RTX-Karte...")
        self.spinner.start("Llama 1B arbeitet...")
        with torch.no_grad():
            outputs = model.generate(**inputs, **config['params'])
        self.spinner.stop()

        # 5. Die ANTWORT
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        token_count = outputs[0].shape[0]

        print(f"[STEP 5] KI-Antwort erhalten ({token_count} Tokens)")
        print(f"{'='*20} 🔍 DEBUG ENDE {'='*20}\n")
        
        # Die Rückgabe füllt automatisch das Feld 'llama_out' im State (wegen Decorator)
        return response, token_count
    # 1. ANALYST: Llama 3.2 3B (Historischer Kontext)
    @log_node_performance("meta-llama/Llama-3.2-3B-Instruct")
    def llama_3_2_3_b_node(self, state):
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        model, tokenizer = self.manager.load_by_name(model_name)
        config = PortfolioPrompts.get_analyst_config(state.get("portfolio_items"))
        
        prompt = f"<|system|>\n{config['system']}\n<|user|>\n{config['user']}\n<|assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.spinner.start("Llama analysiert historische Formen...")
        with torch.no_grad():
            outputs = model.generate(**inputs, **config['params'])
        self.spinner.stop()
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True), outputs[0].shape[0]

    # 2. EXTRAKTOR: Qwen 1.7B (Daten & Prozentwerte)
    @log_node_performance("rd211/Qwen3-1.7B-Instruct")
    def qwen_3_1_7b_node(self, state: Dict[str, Any]):
        model_name = "rd211/Qwen3-1.7B-Instruct"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        # Debugging für Quantisierung
        print(f"DEBUG: Qwen Datentyp: {model.dtype}")
        print(f"DEBUG: Qwen Speicher: {model.get_memory_footprint() / 1024**2:.2f} MB") 
        
        config = PortfolioPrompts.get_qwen_config(state.get("portfolio_items"))
        prompt = f"<|im_start|>system\n{config['system']}<|im_end|>\n<|im_start|>user\n{config['user']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.spinner.start("Qwen extrahiert Länder-Fakten...")
        with torch.no_grad():
            outputs = model.generate(**inputs, **config['params'])
        self.spinner.stop()

        return tokenizer.decode(outputs[0], skip_special_tokens=True), outputs[0].shape[0]

    # 3. REASONER: DeepSeek R1 7B (Ökonomischer Audit)
    @log_node_performance("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    def deepseek_r1_1_5b_node(self, state: Dict[str, Any]): # 🔵 Name exakt für den 1.5B Aufruf
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        # Abgleich der Ergebnisse von Llama und Qwen
        llama_out = state.get('meta-llama/Llama-3.2-3B-Instruct', '')
        qwen_out = state.get('rd211/Qwen3-1.7B-Instruct', '')
        context = f"Historische Analyse: {llama_out}\nDaten-Extraktion: {qwen_out}"
        
        config = PortfolioPrompts.get_deepseek_config(context)
        prompt = f"<|thought|>\n{config['system']}\n{config['user']}"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        self.spinner.start("DeepSeek 1.5B prüft die ökonomische Logik...")
        with torch.no_grad():
            outputs = model.generate(**inputs, **config['params'])
        self.spinner.stop()
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True), outputs[0].shape[0]

    # 4. STRATEGE: Mistral 7B (Politik-Empfehlung)
    @log_node_performance("Mistral-7B-Instruct-v0.3")
    def mistral_7b_node(self, state: Dict[str, Any]):
        model_name = "Mistral-7B-Instruct-v0.3"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        audit_data = state.get("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "")
        config = PortfolioPrompts.get_mistral_config(audit_data)
        
        prompt = f"<s>[INST] {config['system']}\n{config['user']} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.spinner.start("Mistral entwirft Strategie...")
        with torch.no_grad():
            outputs = model.generate(**inputs, **config['params'])
        self.spinner.stop()
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True), outputs[0].shape[0]

    # 5. FORMATIERER: Google Gemma (JSON-Struktur)
    @log_node_performance("google/gemma-3n-E2B-it")
    def gemma_2b_node(self, state: Dict[str, Any]):
        model_name = "google/gemma-3n-E2B-it"
        model, tokenizer = self.manager.load_by_name(model_name)
        
        mistral_out = state.get('Mistral-7B-Instruct-v0.3', '')
        config = PortfolioPrompts.get_gemma_config(mistral_out)
        
        prompt = f"<start_of_turn>model\n{config['system']}<end_of_turn>\n<start_of_turn>user\n{config['user']}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        self.spinner.start("Gemma finalisiert JSON...")
        with torch.no_grad():
            outputs = model.generate(**inputs, **config['params'], )
        self.spinner.stop()

        return tokenizer.decode(outputs[0], skip_special_tokens=True), outputs[0].shape[0]

    # 6. REPORTER (Export)
    def reporter_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n--- Reporter: Exportiere Ergebnisse ---")
        internal_keys = ['portfolio_items', 'metrics', 'report', 'report_complete']
        report_sections = ["# FINALES MIETREGULIERUNGS-DOSSIER\n"]

        for key, value in state.items():
            if key not in internal_keys and isinstance(value, str):
                clean_name = key.split('/')[-1]
                report_sections.append(f"## 🔹 Modell: {clean_name}\n{value}\n\n---")

        full_report = "\n".join(report_sections)
        if self.interface: self.interface.save_all(state)
        return {"report": full_report, "report_complete": True}