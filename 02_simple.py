
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Dict, Any, Annotated, List
from IPython.display import display, Image
import graphviz
from nodes import SpecializedNodes
from model_manager import LocalModelManager
from workflow_logger import WorkflowLogger
from workflow_interface import WorkflowInterface
##ich uebergebe den manager die auswahl der modelle. er verwaltet das laden und entladen
manager = LocalModelManager("modelle.json")
logger = WorkflowLogger()
interface = WorkflowInterface()
nodes = SpecializedNodes(manager, logger, interface)

# class PortfolioState(TypedDict):
#     portfolio_items: list[dict]
#     analyst_out: str    # Ergebnis von Llama
#     teacher_out: str    # Ergebnis von DeepSeek
#     strategist_out: str # Ergebnis von Mistral
#     metrics: list[dict] # Die Performance-Daten
#     report: str         # Der finale Text

class PortfolioState(TypedDict):
    # --- Start-Objekt (Die Quelle) ---
    portfolio_items: List[Dict[str, Any]]
    # --- Zwischenergebnisse (Parallele Nodes) ---
    # Wir nutzen den Funktionsnamen als Key (passend zum state_packer)
    llama_3_2_3_b_node: str  # Analyse von Llama
    qwen_3_1_7b_node: str    # Daten-Extraktion von Qwen
    # --- Kontroll-Ebene ---
    deepseek_r1_7b_node: str # Abgleich & Erkenntnisse
    # --- Struktur-Ebene ---
    gemma_2b_node: Dict[str, Any] # Finales JSON-Format
    # --- Infrastruktur ---
    metrics: Annotated[List[Dict[str, Any]], operator.add]
    report: str
#nodes gestalten
def calc_total(state: PortfolioState) -> PortfolioState:
    state["total_usd"] = state["amount_usd"] * 1.08
    return state

def convert_to_inr(state: PortfolioState) -> PortfolioState:
    state["total"] = state["amount_usd"] * 85
    return state

def convert_to_eur(state: PortfolioState) -> PortfolioState:
    state["total"] = state["amount_usd"] * 0.92
    return state

def choose_conversion(state: PortfolioState) -> str:
    return state["target_currency"]

#nodes definieren
workflow = StateGraph(PortfolioState) #DTO definieren
workflow.add_node("llama323B", nodes.llama_3_2_3_b_node)#16
workflow.add_node("deepseekr17B", nodes.deepseek_r1_7b_node)
workflow.add_node("mistral7B", nodes.mistral_7b_node)#30
workflow.add_node("gemma2B", nodes.gemma_2b_node)
workflow.add_node("deepseekR115B", nodes.deepseek_r1_1_5b_node)#35
workflow.add_node("qwen317B", nodes.qwen_3_1_7b_node)#11
workflow.add_node("reporter", nodes.reporter_node)

# Kanten definieren
#nodes verbinden und flow definieren
# --- FLOW DEFINITION ---

# 1. Fan-out: Beide starten gleichzeitig aus der Quelle
workflow.add_edge(START, "llama323B")
workflow.add_edge(START, "qwen317B")

# 2. Fan-in: DeepSeek wartet, bis BEIDE fertig sind
workflow.add_edge("llama323B", "deepseekr17B")
workflow.add_edge("qwen317B", "deepseekr17B")

# 3. Lineare Weitergabe
workflow.add_edge("deepseekr17B", "gemma2B")
workflow.add_edge("gemma2B", "reporter")
workflow.add_edge("reporter", END)

workflow.add_edge(START, "llama323B")
workflow.add_edge(START, "qwen317B")
workflow.add_edge("llama323B", "deepseekr17B")
workflow.add_edge("deepseekr17B", "gemma2B")
workflow.add_edge("gemma2B", "reporter")
workflow.add_edge("reporter", END)


#workflow.add_edge("mistral7B", "gemma2B")
#workflow.add_edge("gemma2B", "deepseekR115B")
#workflow.add_edge("deepseekR115B", "qwen317B")
#workflow.add_edge("qwen317B", "reporter")
# workflow.add_edge("reporter", END)


# workflow.add_edge(START, "analyst")
# workflow.add_edge("analyst", "teacher")
# workflow.add_edge("teacher", "strategist")
# workflow.add_edge("strategist", "reporter")
# workflow.add_edge("reporter", END)
#nodes verbinden und flow definieren

graph = workflow.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))
print(graph.get_graph().draw_ascii())

initial_input = {
    "portfolio_items": [
        {"name": "Nvidia", "amount": 10},
        {"name": "Bitcoin", "amount": 0.5},
        
    ],
    "metrics": [] # Wichtig, damit die Liste existiert!
}
print("Starte Workflow...")
final_state = graph.invoke(initial_input)
# display(Image(graph.get_graph().draw_mermaid_png()))
print(graph.get_graph().draw_ascii())

# def portfolio_flow(amount_usd: float) -> dict:
#     return graph.run({"amount_usd": amount_usd})

print("\n--- FINALER BERICHT ---")
print(final_state["report"])