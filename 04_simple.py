from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict, Any, Optional
from nodes import SpecializedNodes
from model_manager import LocalModelManager, get_ts
from workflow_logger import WorkflowLogger
from workflow_interface import WorkflowInterface
from workflow_utils import state_packer
from typing import Annotated
import operator
# --- 1. INITIALISIERUNG ---
# Wir laden alle bereitgestellten Funktionen
manager = LocalModelManager("modelle.json")
logger = WorkflowLogger()       
interface = WorkflowInterface()
nodes = SpecializedNodes(manager, logger, interface)

# --- 2. STATE DEFINITION ---
class PortfolioState(TypedDict, total=False):  # total=False macht den State flexibel
    portfolio_items: Annotated[list, operator.add]
    # Die Keys müssen zu den Decorators in nodes.py passen (Model-Namen)
    # Wenn deine Nodes die Namen als Keys nutzen, lassen wir das Dict flexibel.
    metrics: List[Dict[str, Any]]
    report: str
    report_complete: bool

# --- 3. NODES GESTALTEN ---
# Die alten Währungs-Funktionen passen nicht zum PortfolioState TypedDict
# und würden Fehler werfen. Wir konzentrieren uns auf die KI-Nodes.

# --- 4. GRAPH DEFINITION ---
workflow = StateGraph(Annotated[dict, operator.ior])

# Alle bereitgestellten Nodes registrieren
workflow.add_node("llama", state_packer(nodes.llama_3_2_3_b_node))
workflow.add_node("qwen", state_packer(nodes.qwen_3_1_7b_node))
workflow.add_node("strategist", state_packer(nodes.mistral_7b_node))
#workflow.add_node("teacher", nodes.deepseek_r1_7b_node)
# workflow.add_node("risk_check", nodes.gemma_2b_node)
# workflow.add_node("logic_check", nodes.deepseek_r1_1_5b_node)

workflow.add_node("reporter", nodes.reporter_node)


# --- 5. FLOW DEFINIEREN (Kanten) ---
# 🔵 Wir bauen eine stabile Kette, die deine 12GB VRAM schont
workflow.add_edge(START, "llama")
workflow.add_edge("llama", "strategist")
workflow.add_edge("strategist", "qwen")
workflow.add_edge("qwen", "reporter")
workflow.add_edge("reporter", END)

# Kompilieren
graph = workflow.compile()

# --- 6. EXECUTION ---
if __name__ == "__main__":
    # Wir bereiten die Eingabe so vor, dass die Nodes damit arbeiten können
    initial_input = {
        "portfolio_items": [
            {"name": "Nvidia", "amount": 10},
            {"name": "Bitcoin", "amount": 0.5},
            {"name": "Apple", "amount": 50}
        ],
        "metrics": [] 
    }

    print(graph.get_graph().draw_ascii())
    print(f"[{get_ts()}]  Starte Workflow...")

    try:
        # 🔵 Der Workflow startet und nutzt jetzt den echten WorkflowLogger
        final_state = graph.invoke(initial_input)

        print("\n" + "="*60)
        print("---  FINALER BERICHT ---")
        # 🔵 Falls der Reporter "report" zurückgibt, drucken wir ihn hier
        print(final_state.get("report", "Kein Bericht generiert."))
        print("="*60)

    except Exception as e:
        print(f"\n FEHLER im Workflow: {e}")