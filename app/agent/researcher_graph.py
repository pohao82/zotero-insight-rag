from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class ResearchState(TypedDict):
    question: str
    context: str
    draft: str
    feedback: str
    iterations: int
    verified: bool

# Wrap Classes into Nodes
# _node(state, func/obj)
def draft_node(state: ResearchState, generator):
    # Calls your existing .draft() method
    answer = generator.draft(state["question"], state["context"])
    return {"draft": answer, "iterations": state.get("iterations", 0) + 1}

def critique_node(state: ResearchState, critic):
    # Calls your existing .verify() method
    feedback = critic.verify(state["draft"], state["context"])
    is_passed = "PASSED" in feedback.upper()
    return {"feedback": feedback, "verified": is_passed}

def refine_node(state: ResearchState, generator):
    # Calls your existing .refine_draft() method
    refined = generator.refine_draft(state["draft"], state["feedback"])
    return {"draft": refined, "iterations": state["iterations"] + 1}

# construct workflow graph
def create_research_graph(generator, critic, max_retries):
    workflow = StateGraph(ResearchState)

    # 1. Add Nodes (Passing your class instances into the functions)
    workflow.add_node("drafter", lambda state: draft_node(state, generator))
    workflow.add_node("critic", lambda state: critique_node(state, critic))
    workflow.add_node("refiner", lambda state: refine_node(state, generator))

    # 2. Define Edges (The Logic Flow)
    workflow.add_edge(START, "drafter")

    # New Router function to decide: Critic or END?
    def route_after_draft(state):
        if max_retries == 0:
            return END
        return "critic"

    # Route from drafter based on max_retries
    workflow.add_conditional_edges("drafter", route_after_draft)
    #workflow.add_edge("drafter", "critic")

    # 3. Add Conditional Routing (The Reflection Loop)
    def should_continue(state):
        if state["verified"] or state["iterations"] > max_retries:
            return END
        return "refiner"

    workflow.add_conditional_edges("critic", should_continue)
    workflow.add_edge("refiner", "critic")

    return workflow.compile()
