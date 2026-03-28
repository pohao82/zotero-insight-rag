from app.core.config import create_research_engine, retriever_module

def main(question):

    #reflection_loop, _ = create_research_engine()
    max_retries=0
    graph_loop, _ = create_research_engine(overrides=None)
    retriever = retriever_module()

    top_k=5
    window=2
    _, sources, context_map = retriever.get_relevant_context(question, top_k=top_k, window=window)

    context = ""
    for source_i, context_i in context_map.items():
        context += f'\n{source_i}\n{context_i}'
    print(context)

    # langGraph
    #  ResearchState(TypedDict):
    initial_input = {
        "question": question, 
        "context": context,
        "iterations": 0,
        "verified": False,
        "feedback": "",
        "max_retries": max_retries
    }

    # Invoke the graph (returns the final dictionary of the State)
    final_state = graph_loop.invoke(initial_input)

    # Extract the specific data you need
    answer = final_state["draft"]
    verified = final_state["verified"]
    feedback = final_state["feedback"]


    # print only cited contexts
    import re
    regex = r'([\[【(]Source\d+[\]】)])' # get the entire tag [Source1]
    cited_tags = re.findall(regex, answer)
    print(cited_tags)

    cited_context = ""
    if len(cited_tags) > 0:
        for c in cited_tags:
            cited_context += context_map[c]
            answer = answer.replace(c,f'\n\n{c}:\n{context_map[c]}')
    #

    print(f"\n\n[{'✅' if verified else '⚠️'}] ANSWER:\n{answer} \n FEEDBACK:\n {feedback}")

if __name__ == "__main__":
    # example question
    question = "lattice parameters of La2O3Mn2Se2 at temperature at 200 K, 150K, 100 K, 6 K"
    print(f"[*] Retrieving physics context for: {question}")
    main(question)
