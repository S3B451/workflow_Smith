import json
import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

class LocalModelManager:
    def __init__(self, json_path: str):
        self.active_path = None
        self.model = None
        self.tokenizer = None
        self.processor = None # Für Vision-Modelle (Phi, Florence, etc.)
        
        # JSON einlesen und Mapping erstellen
        with open(json_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)
        
        # Schneller Zugriff über den Namen
        self.model_lookup = {m['name']: m['path'] for m in self.model_data}

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

        print(f"--- [Manager] Lade: {model_path} ---")
        
        # Unterscheidung Vision vs. Text
        if is_vision:
            # Für Modelle wie Florence-2 oder Phi-3.5-vision
            from transformers import AutoModelForVision2Seq
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
        
        self.active_path = model_path
        return self.model, (self.tokenizer if not is_vision else self.processor)

    def unload(self):
        print(f"--- [Manager] Entlade VRAM ---")
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.active_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()