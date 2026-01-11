
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from IPython.display import display, Image
import graphviz

class PortfolioState(TypedDict):
    amount_usd: float
    target_currency: Literal["INR", "EUR"]
    total_usd: float
    total: float
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
builder = StateGraph(PortfolioState) #DTO definieren
builder.add_node("calc_total", calc_total)
builder.add_node("convert_to_inr", convert_to_inr)
builder.add_node("convert_to_eur", convert_to_eur)

#nodes verbinden und flow definieren
builder.add_edge(START, "calc_total")
builder.add_conditional_edges(
    "calc_total",
     choose_conversion,
     {
         "INR": "convert_to_inr", 
         "EUR": "convert_to_eur"
         }) #input und funktion als output
builder.add_edge("convert_to_inr", END)
builder.add_edge("convert_to_eur", END)

graph = builder.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))
print(graph.get_graph().draw_ascii())

# def portfolio_flow(amount_usd: float) -> dict:
#     return graph.run({"amount_usd": amount_usd})

# result = portfolio_flow(1000)
# print("Final State:", result)