from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict, Any, Annotated
import operator

# Deine Module importieren
from nodes import SpecializedNodes
from model_manager import LocalModelManager
from workflow_logger import WorkflowLogger
from workflow_interface import WorkflowInterface
from prompt_library import PortfolioPrompts

# --- 1. STATE DEFINITION ---
class PortfolioState(TypedDict):
    portfolio_items: Any # Kann Text oder Pfad sein
    # Die Keys MÜSSEN exakt so heißen wie die Modell-IDs im Logger/Decorator
    # damit die Daten dort landen.
    llama_out: str 
    qwen_out: str
    deepseek_out: str
    gemma_out: str
    metrics: Annotated[List[Dict[str, Any]], operator.add]
    report: str
    report_complete: bool

# --- 2. INITIALISIERUNG ---
manager = LocalModelManager("modelle.json")
logger = WorkflowLogger()
interface = WorkflowInterface()
nodes = SpecializedNodes(manager, logger, interface)

# --- 3. GRAPH AUFBAU (Fan-out / Fan-in) ---
workflow = StateGraph(PortfolioState)

# # Nodes hinzufügen
# #workflow.add_node("qwen_node", nodes.qwen_3_1_7b_node)
# workflow.add_node("deepseek_node", nodes.deepseek_r1_1_5b_node)
# workflow.add_node("mistral_node", nodes.mistral_7b_node)
# workflow.add_node("reporter", nodes.reporter_node)
# workflow.add_node("llama_node", nodes.llama_3_2_3_b_node)
# # --- DER FLOW (Edges) ---
# workflow.add_edge(START, "mistral_node")
# workflow.add_edge("mistral_node", "deepseek_node")
# workflow.add_edge("deepseek_node", "llama_node")
# workflow.add_edge("llama_node", "reporter")
# workflow.add_edge("reporter", END)



workflow.add_node("test_node", nodes.llama_1b_test_node)
workflow.add_node("reporter", nodes.reporter_node)

# Einfacher Flow
workflow.add_edge(START, "test_node")
workflow.add_edge("test_node", "reporter")
workflow.add_edge("reporter", END)




# # Parallelstart (Fan-out)
# workflow.add_edge(START, "llama323B")
# workflow.add_edge(START, "qwen317B")

# # Zusammenführung (Fan-in): DeepSeek wartet auf BEIDE
# workflow.add_edge("llama323B", "deepseekr17B")
# workflow.add_edge("qwen317B", "deepseekr17B")

# # Finale Kette
# workflow.add_edge("deepseekr17B", "gemma2B")
# workflow.add_edge("gemma2B", "reporter")
# workflow.add_edge("reporter", END)

graph = workflow.compile()

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # Du kannst hier Text oder einen Pfad übergeben
    initial_input = {
        "portfolio_items": PortfolioPrompts.DEFAULT_PORTFOLIO,
        "metrics": []
    }

    print(f"\n Starte parallelen Workflow...")
    print(graph.get_graph().draw_ascii()) # Zeigt dir die Struktur in der Konsole
    
    final_state = graph.invoke(initial_input)

    print("\n" + "="*50)
    print(" WORKFLOW ABGESCHLOSSEN")
    print("="*50)
    print(final_state.get("report", "Fehler: Kein Bericht generiert."))