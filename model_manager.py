import json
import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

class LocalModelManager:
    def __init__(self, json_path: str):
        self.active_path = None # Aktuell geladenes Modell
        self.model = None
        self.tokenizer = None
        self.processor = None # Für Vision-Modelle (Phi, Florence, etc.)
        
        # JSON einlesen und Mapping erstellen
        with open(json_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)
        
        # Schneller Zugriff über den Namen
        self.model_lookup = {m['name']: m['path'] for m in self.model_data}

    ############################### GPU Ladeanzeige ###############################################
    def _get_vram_info(self):
        """Hilfsfunktion: Gibt freien und totalen VRAM in MB zurück."""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1024**2, total / 1024**2
        return 0, 0

    def _print_stats(self, message):
        """Druckt eine Statusmeldung mit aktuellen VRAM Werten."""
        free, total = self._get_vram_info()
        used = total - free
        print(f"--- [{message}] VRAM: {used:.0f}MB / {total:.0f}MB genutzt ({free:.0f}MB frei) ---")

    ###############################################################################################

    def load_by_name(self, model_name: str, is_vision: bool = False):
        if model_name not in self.model_lookup:
            raise ValueError(f"Modell '{model_name}' nicht in der JSON gefunden!")
        
        model_path = self.model_lookup[model_name]
        return self._load_from_path(model_path, is_vision)

    def _load_from_path(self, model_path: str, is_vision: bool):
        if self.active_path == model_path:
            return self.model, self.tokenizer if not is_vision else self.processor

        if self.active_path is not None:
            self.unload()

        self._print_stats("Vor Laden")
        print(f"--- [Manager] Lade: {model_path} ---")
        
        # Unterscheidung Vision vs. Text
        if is_vision:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
            # Florence-2 mag kein device_map="auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                dtype=torch.float16,    # "dtype" statt "torch_dtype" (sauberer)
                trust_remote_code=True,
                attn_implementation="eager"
                # device_map="auto" <-- Diese Zeile muss weg!
            ).to("cuda") # <-- Das hier explizit hinzufügen
            self.tokenizer = None
            
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
        
        self.active_path = model_path
        self._print_stats("Nach Laden")
        return self.model, (self.tokenizer if not is_vision else self.processor)

    def unload(self):
        self._print_stats("Vor Entladen")
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.active_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self.active_path = None
        self._print_stats("Nach Entladen")



    