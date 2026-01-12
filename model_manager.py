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

    def load_by_name(self, model_name: str):
        if model_name not in self.model_lookup:
            raise ValueError(f"Modell '{model_name}' nicht in der JSON gefunden!")
        
        model_path = self.model_lookup[model_name]
        return self._load_from_path(model_path)

    def _load_from_path(self, model_path: str):
        if self.active_path == model_path:
            return self.model, self.tokenizer

        if self.active_path is not None:
            self.unload()

        print(f"[{get_ts()}] [VRAM] LOAD start: {model_path}") 
        self.spinner.start("Lade Modellgewichte...")
        start_load = time.time() 
        try:    
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                use_fast=False
            )
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype= "auto", 
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ KRITISCHER LADEFEHLER: {str(e)}")
            raise
        finally:
            self.spinner.stop() 

        load_duration = time.time() - start_load 
        free_gb, total_gb = self._get_vram_info() 
        used_gb = total_gb - free_gb 
        
        self.active_path = model_path
        
        print(f"[{get_ts()}] [VRAM] LOAD finished in {load_duration:.2f}s | Occupied: {used_gb:.2f} GB") 
        return self.model, self.tokenizer 

    def unload(self):
        """Leert den VRAM so gründlich wie möglich."""
        if self.active_path is None:
            return      
        print(f"[{get_ts()}] [VRAM] UNLOAD start") 
        start_unload = time.time() 
        
        self.model = None
        self.tokenizer = None
        self.active_path = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        unload_duration = time.time() - start_unload 
        free_gb, total_gb = self._get_vram_info() 
        
        
        print(f"[{get_ts()}] [VRAM] UNLOAD finished in {unload_duration:.2f}s | Free: {free_gb:.2f} GB") 