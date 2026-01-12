
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
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

class PortfolioState(TypedDict):
    portfolio_items: list[dict]
    analyst_out: str    # Ergebnis von Llama
    teacher_out: str    # Ergebnis von DeepSeek
    strategist_out: str # Ergebnis von Mistral
    metrics: list[dict] # Die Performance-Daten
    report: str         # Der finale Text
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
workflow.add_node("analyst", nodes.llama_3_2_3_b_node)
workflow.add_node("teacher", nodes.deepseek_r1_7b_node)
workflow.add_node("strategist", nodes.mistral_7b_node)
workflow.add_node("reporter", nodes.reporter_node)

# Kanten definieren
#nodes verbinden und flow definieren

workflow.set_entry_point("analyst")
workflow.add_edge("analyst", "teacher")
workflow.add_edge("teacher", "strategist")
workflow.add_edge("strategist", "reporter")
workflow.add_edge("reporter", END)
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