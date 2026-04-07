from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class ResearchState(TypedDict):
    question: str
    context: str
    draft: str
    feedback: str
    iterations: int
    verified: bool
    max_retries: int
    # Track history 
    #full_history: List[str]

# Wrap existing lecl chain into nodes
# _node(state, func/obj)
# ChatPromptTemplate  defined in modular
def draft_node(state: ResearchState, generator):
    answer = generator.draft_chain.invoke({
        "question": state["question"], 
        "context": state["context"]
    })
    return {"draft": answer, "iterations": state.get("iterations", 0) + 1}

def critique_node(state: ResearchState, critic):
    feedback = critic.verify_chain.invoke({
        "draft": state["draft"], 
        "context": state["context"]
    })
    is_passed = "PASSED" in feedback.upper()
    return {"feedback": feedback, "verified": is_passed}

def refine_node(state: ResearchState, generator):
    refined = generator.refine_chain.invoke({
        "draft": state["draft"], 
        "feedback": state["feedback"]
    })
    return {"draft": refined, "iterations": state["iterations"] + 1}

# Construct workflow graph
def create_research_graph(generator, critic):
    workflow = StateGraph(ResearchState)

    # 1. Add Nodes (Passing the class instances into the functions)
    workflow.add_node("drafter", lambda state: draft_node(state, generator))
    workflow.add_node("critic", lambda state: critique_node(state, critic))
    workflow.add_node("refiner", lambda state: refine_node(state, generator))

    # 2. Define Edges (The Logic Flow)
    workflow.add_edge(START, "drafter")

    # New Router function to decide: Critic or END?
    def route_after_draft(state: ResearchState):
        if state.get("max_retries", 0) == 0:
            return END
        return "critic"

    # Route from drafter based on max_retries
    workflow.add_conditional_edges("drafter", route_after_draft)

    # 3. Add Conditional Routing (The Reflection Loop)
    def should_continue(state: ResearchState):
        if state["verified"] or state["iterations"] > state["max_retries"]:
            return END
        return "refiner"

    workflow.add_conditional_edges("critic", should_continue)
    workflow.add_edge("refiner", "critic")

    return workflow.compile()
