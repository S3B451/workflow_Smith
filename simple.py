
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import display, Image
import graphviz

class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    total_inar: float

def calc_total(state: PortfolioState) -> PortfolioState:
    state["total_inar"] = state["amount_usd"] * 1.08
    return state

def convert_to_inr(state: PortfolioState) -> PortfolioState:
    state["total_inar"] = state["amount_usd"] * 85
    return 

builder = StateGraph(PortfolioState)

builder.add_node("calc_total", calc_total)
builder.add_node("convert_to_inr", convert_to_inr)

builder.add_edge(START, "calc_total")
builder.add_edge("calc_total", "convert_to_inr")
builder.add_edge("convert_to_inr", END)

graph = builder.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))
print(graph.get_graph().draw_ascii())

# def portfolio_flow(amount_usd: float) -> dict:
#     return graph.run({"amount_usd": amount_usd})

# result = portfolio_flow(1000)
# print("Final State:", result)