from typing import Dict, Any
import torch



class SpecializedNodes:
    def __init__(self, manager):
        self.manager = manager

    def analyst_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node für die Portfolio-Analyse."""
        print("--- Node: Analyst wird aktiv ---")
        
        # Modell laden (Pfad anpassen!)
        model, tokenizer = self.manager.load("meta-llama_Llama-3.2-3B-Instruct")
        
        prompt = f"Analysiere folgendes Portfolio auf Risiko: {state.get('portfolio_items')}"
        
        # Einfacher Inference-Block
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"analysis": response}
    
    def vision_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node für die Portfolio-Optimierung."""
        print("--- Node: Teacher wird aktiv ---")
        
        model, processor = self.manager.load("microsoft/Florence-2-large", is_vision=True)
        
        prompt = f"Optimiere folgendes Portfolio: {state.get('portfolio_items')}"
        
        inputs = processor(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        
        response = processor.batch_decode(outputs[0], skip_special_tokens=True)
        
        return {"vision_node": response}
    
    def le_french(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node für die Portfolio-Analyse."""
        print("--- Node: le_french wird aktiv ---")
        
        # Modell laden (Pfad anpassen!)
        model, tokenizer = self.manager.load("Ministral-3-3B")
        
        prompt = f"Analysiere folgendes Portfolio auf Risiko: {state.get('portfolio_items')}"
        
        # Einfacher Inference-Block
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"le_french": response}
    
    def reporter_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node, der die Ergebnisse formatiert."""
        print("--- Node: Reporter wird aktiv ---")
        
        return {"report": f"Bericht erstellt basierend auf: {state.get('analysis')[:50]}..."}