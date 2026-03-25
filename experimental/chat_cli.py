from app.core.config import create_research_engine, retriever_module
from app.utils.memory import ChatMemory
import re

# Chatbot with memory
def start_chat(has_memory=False,max_retries=0):
    print("\n=== Physics Research Assistant ===")

    chat_memory = ChatMemory(window_size=0)
    graph_loop, generator = create_research_engine(overrides=None, max_retries=max_retries)
    retriever = retriever_module()

    while True:
        try:
            user_input = input("\nQuery > ").strip()
            if user_input.lower() in ['exit', 'quit']: break

            # get history (if exist) to the query
            chat_history = chat_memory.get_formatted_history()
            print('------- chat_history -------')
            print(chat_history)

            search_query = generator.rewrite_query(user_input, chat_history)
            if search_query != user_input:
                print(f"[*] Searching for: {search_query}")

            # Standard RAG Flow
            metadata_filters = None
            context, sources, context_map = retriever.get_relevant_context(
                user_input, top_k=5, window=2, filter_dict=metadata_filters, mode="standard"
            )

            #verified = False
            #answer, verified = loop.run(search_query, context, max_retries)

            # langGraph
            #  ResearchState(TypedDict):
            initial_input = {
                "question": search_query, 
                "context": context,
                "iterations": 0,
                "verified": False,
                "feedback": ""
            }

            # Invoke the graph (returns the final dictionary of the State)
            final_state = graph_loop.invoke(initial_input)

            # Extract the specific data you need
            answer = final_state["draft"]
            verified = final_state["verified"]
            #feedback = final_state["feedback"]

            regex = r'([\[【(]Source\d+[\]】)])'
            cited_tags = re.findall(regex, answer)
            cited_tags = list(set(cited_tags))
            cited_context = "".join([f"{c}:\n{context_map.get(c, "")}\n\n" for c in cited_tags])

            # Update Memory
            if has_memory:
                chat_memory.add("user", user_input)
                chat_memory.add("assistant", answer)

            if max_retries==0:
                badge = "🔵 No critic:"
            elif verified:
                badge = "✅ **Verified**"
            else:
                badge = "⚠️ **Unverified**"

            # Display
            print(f"\n[{badge}] ANSWER:\n{answer} REFERENCES:\n{cited_context}\n\n")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    start_chat(has_memory=False,max_retries=0)
