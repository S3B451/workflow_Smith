import time
import torch
from datetime import datetime

def get_ts(): 
    return datetime.now().strftime("%H:%M:%S.%f")[:-3] 

class WorkflowLogger:
    def log_step(self, model_name: str, duration: float, token_count: int):
        # VRAM Messung
        vram_used = torch.cuda.memory_allocated() / 1024**3 # Umrechnung in MB
        vram_reserved = torch.cuda.memory_reserved() / 1024**2
        
        # Speed Berechnung
        tps = token_count / duration if duration > 0 else 0
        vram_used_gb = 0.0 
        if torch.cuda.is_available(): 
            vram_used_gb = torch.cuda.memory_allocated()/ 1024**3
        # Konsolen-Ausgabe (Schick formatiert)
        print(f"\n" + "="*50)
        print(f"[{get_ts()}] [PERF] INFERENCE: {model_name}") 
        print(f"[{get_ts()}] [PERF] Duration: {duration:.2f}s | Speed: {tps:.2f} t/s | Tokens: {token_count}") 
        print(f"[{get_ts()}] [PERF] VRAM belegt: {vram_used_gb:.2f} GB") 
        print("="*50 + "\n")
        
        # Rückgabe als Dictionary für das DTO (State)
        return {
            "model": model_name,
            "duration_sec": round(duration, 2),
            "tokens": token_count,
            "speed_tps": round(tps, 2),
            "vram_mb": round(vram_used, 0)
        }