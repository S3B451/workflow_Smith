import os
import sys
import time
venv_site_packages = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")

# 2. Wir schieben diesen Pfad an die ALLERERSTE Stelle der Suchliste
if os.path.exists(venv_site_packages):
    sys.path.insert(0, venv_site_packages)
    print(f"---  PFAD-ANKER GESETZT: Nutze .venv aus {venv_site_packages} ---")
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, START, END

# Importiere deine Komponenten
from nodes import SpecializedNodes
from model_manager import LocalModelManager
from workflow_logger import WorkflowLogger
from workflow_interface import WorkflowInterface

# 1. Setup der Infrastruktur
manager = LocalModelManager("modelle.json")
logger = WorkflowLogger()
interface = WorkflowInterface()
nodes = SpecializedNodes(manager, logger, interface)

# 2. State Definition (DTO)
class PortfolioState(TypedDict):
    portfolio_items: List[dict]
    metrics: List[dict]
    report: str
    # Dynamische Keys für die Modell-Antworten (werden von Decorator befüllt)
    analyst_out: str
    teacher_out: str
    strategist_out: str

# 3. Test-Konfiguration
# Liste aller Nodes, die wir einzeln testen wollen
# Format: (Node_ID_im_Graph, Funktion_in_SpecializedNodes, Anzeigename)

tests = [
    ("analyst", nodes.llama_3_2_3_b_node, "Llama 3.2 3B"),
    #("optimizer", nodes.qwen_2_5_7B_node, "Qwen 2.5 7B"),
    # ("teacher", nodes.deepseek_r1_7b_node, "DeepSeek R1 7B"),
    ("strategist", nodes.mistral_7b_node, "Mistral 7B v0.3"),
    #("risk_check", nodes.gemma_2b_node, "Gemma 2B"),
    #("logic_check", nodes.deepseek_r1_1_5b_node, "DeepSeek R1 1.5B"),
    ("fast_analyst", nodes.qwen_3_1_7b_node, "Qwen 3 1.7B")
]

def run_single_test(node_id, node_func, label):
    print(f"\n{'-'*60}")
    print(f" STARTE EINZELTEST FÜR: {label}")
    print(f"{'-'*60}")
    
    # Minimalen Graphen für diesen Test bauen
    test_workflow = StateGraph(PortfolioState)
    test_workflow.add_node(node_id, node_func)
    test_workflow.add_node("reporter", nodes.reporter_node)
    test_workflow.add_edge(START, node_id)
    test_workflow.add_edge(node_id, "reporter")
    test_workflow.add_edge("reporter", END)
    
    test_graph = test_workflow.compile()
    
    # Test-Input
    initial_input = {
        "portfolio_items": [{"name": "Nvidia", "amount": 10}, {"name": "Bitcoin", "amount": 0.5}],
        "metrics": []
    }
    
    try:
        start_ts = time.time()
        result = test_graph.invoke(initial_input)
        duration = time.time() - start_ts
        
        print(f"\n TEST ERFOLGREICH ({duration:.2f}s)")
        print(f"Bericht-Vorschau: {result.get('report', 'Kein Bericht erhalten')[:200]}...")
        
    except Exception as e:
        print(f"\n FEHLER BEI {label}:")
        print(f"Details: {str(e)}")

# 4. Hauptschleife: Alle Tests nacheinander ausführen
if __name__ == "__main__":
    print(f"Starte Test-Suite für {len(tests)} Modelle...")
    
    for node_id, node_func, label in tests:
        run_single_test(node_id, node_func, label)
        # Kurze Pause für VRAM-Flush
        time.sleep(1)
        
    print(f"\n{'='*60}")
    print("ALLE TESTS ABGESCHLOSSEN")
    print(f"Prüfe die Logs in: {interface.dAbA_dir}")
    print(f"{'='*60}")