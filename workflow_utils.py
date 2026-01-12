# from typing import Any, Dict, Callable

# def state_packer(node_func: Callable):
#     """
#     Dieser Adapter sorgt dafür, dass JEDE Node automatisch 
#     ihre Ergebnisse in den LangGraph-State schreibt.
#     """
#     def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
#         # 1. Führe die KI-Node aus
#         result = node_func(state)
        
#         # 2. Extrahiere den Text (falls ein Tupel (Text, Tokens) kommt)
#         content = result[0] if isinstance(result, tuple) else result
        
#         # 3. Bestimme den Speichernamen (Name der Funktion)
#         storage_key = node_func.__name__
        
#         # DEBUG-Meldung in der Konsole
#         print(f"   [STATE] Speichere Ergebnis für '{storage_key}'...")
        
#         # 4. Rückgabe als Dictionary (Zwingend für LangGraph!)
#         return {storage_key: content}
    
#     # Den Namen beibehalten, damit LangGraph nicht verwirrt ist
#     wrapper.__name__ = node_func.__name__
#     return wrapper

def state_packer(node_func):
    def wrapper(state):
        result = node_func(state)
        # Handle Tupel (Text, Tokens)
        content = result[0] if isinstance(result, tuple) else result
        
        storage_key = node_func.__name__
        
        # 🔵 DEBUG: Wir drucken die ersten 20 Zeichen des Inhalts
        print(f"   [STATE] -> {storage_key} schreibt: {str(content)[:20]}...")
        
        return {storage_key: content}
    
    wrapper.__name__ = node_func.__name__
    return wrapper