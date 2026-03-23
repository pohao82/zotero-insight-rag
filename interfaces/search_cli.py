from app.core.config import retriever_module

# Chatbot with memory
def start_chat(top_k=5,window=3):
    print("\n=== Physics Research Assistant (with Memory) ===")

    retriever = retriever_module()

    while True:
        try:
            user_input = input("\nQuery > ").strip()
            if user_input.lower() in ['exit', 'quit']: break

            # Standard RAG Flow
            metadata_filters = None
            context, sources, context_map = retriever.get_relevant_context(
                user_input, top_k, window, filter_dict=metadata_filters, mode="standard"
            )

            cited_context = ""
            break_line = '-'*70
            for source_tag in context_map:
                cited_context += '\n\n'+break_line
                cited_context += f'\n{source_tag}\n'
                cited_context += break_line
                cited_context += context_map[source_tag]

            # Display
            print(f"ANSWER:\n{cited_context}")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    start_chat()
