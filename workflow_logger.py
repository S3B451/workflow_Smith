import time
import torch
from datetime import datetime

class WorkflowLogger:
    def log_step(self, model_name: str, duration: float, token_count: int):
        # VRAM Messung
        vram_used = torch.cuda.memory_allocated() / 1024**2  # Umrechnung in MB
        vram_reserved = torch.cuda.memory_reserved() / 1024**2
        
        # Speed Berechnung
        tps = token_count / duration if duration > 0 else 0
        
        # Konsolen-Ausgabe (Schick formatiert)
        print(f"\n" + "="*50)
        print(f"METRIKEN - {model_name}")
        print(f" Zeit:   {duration:.2f} s")
        print(f"Tokens: {token_count}")
        print(f"Speed:  {tps:.2f} tokens/s")
        print(f"VRAM:   {vram_used:.0f} MB (belegt) / {vram_reserved:.0f} MB (reserviert)")
        print("="*50 + "\n")
        
        # Rückgabe als Dictionary für das DTO (State)
        return {
            "model": model_name,
            "duration_sec": round(duration, 2),
            "tokens": token_count,
            "speed_tps": round(tps, 2),
            "vram_mb": round(vram_used, 0)
        }