import json
import os
import torch
import gc
import time 
from datetime import datetime 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from console_feedback import ActivitySpinner

def get_ts(): 
    return datetime.now().strftime("%H:%M:%S.%f")[:-3] 

class LocalModelManager:
    def __init__(self, json_path: str):
        self.spinner = ActivitySpinner()
        self.active_path = None 
        self.model = None
        self.tokenizer = None
        self.processor = None 
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)
        
        self.model_lookup = {m['name']: m['path'] for m in self.model_data}

    ############################### GPU Ladeanzeige ###############################################
    def _get_vram_info(self):
        """Hilfsfunktion: Gibt freien und totalen VRAM in GB zurück.""" 
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1024**3, total / 1024**3 
        return 0, 0

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

        print(f"[{get_ts()}] [VRAM] LOAD start: {model_path}") 
        self.spinner.start("Lade Modellgewichte...")
        start_load = time.time() 
        
        if is_vision:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                dtype=torch.float16, 
                trust_remote_code=True,
                attn_implementation="eager"
            ).to("cuda")
            self.tokenizer = None
            
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
        self.spinner.stop()    
        load_duration = time.time() - start_load 
        free_gb, total_gb = self._get_vram_info() 
        used_gb = total_gb - free_gb 
        
        self.active_path = model_path
        # 🔵 self._print_stats("Nach Laden")
        print(f"[{get_ts()}] [VRAM] LOAD finished in {load_duration:.2f}s | Occupied: {used_gb:.2f} GB") 
        return self.model, (self.tokenizer if not is_vision else self.processor)

    def unload(self):
        # 🔵 self._print_stats("Vor Entladen")
        print(f"[{get_ts()}] [VRAM] UNLOAD start") 
        start_unload = time.time() 
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.active_path = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        unload_duration = time.time() - start_unload 
        free_gb, total_gb = self._get_vram_info() 
        
        # 🔵 self._print_stats("Nach Entladen")
        print(f"[{get_ts()}] [VRAM] UNLOAD finished in {unload_duration:.2f}s | Free: {free_gb:.2f} GB") 