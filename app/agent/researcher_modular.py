from langchain_core.messages import HumanMessage, SystemMessage
from app.agent import prompts
#import re

# The Generator (Drafting & Refining)
class ResearchGenerator:
    def __init__(self, llm):
        self.llm = llm

    def draft(self, question, context):
        return self.llm.invoke([
            SystemMessage(content=prompts.GENERATOR_SYSTEM),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]).content

    def refine_draft(self, draft, feedback):
        return self.llm.invoke([
            SystemMessage(content=prompts.CRITIC_SYSTEM),
            HumanMessage(content=f"Your previous answer had issues: Below is the feedback: {feedback}\n\n to  your original draft: {draft}\n\nRewrite it correctly.")
        ]).content

    # add chat memory for cli
    def rewrite_query(self, user_input, chat_history):
        """Turns follow-up questions into standalone search queries."""
        if not chat_history:
            return user_input

        prompt = f"""Given the following chat history and a follow-up question,
        rephrase the follow-up question to be a standalone search query.

        History: {chat_history[-2:]}
        Follow-up: {user_input}
        Standalone Query:"""

        return self.llm.invoke(prompt).content


# The Critic (Verification)
class ResearchCritic:
    def __init__(self, llm):
        self.llm = llm

    def verify(self, draft, context):
        user_content = prompts.CRITIC_USER_TEMPLATE.format(
            context=context,
            draft=draft
        )

        # Critics often work better with 0.0 temperature (deterministic)
        return self.llm.invoke([
            SystemMessage(content=prompts.CRITIC_SYSTEM),
            HumanMessage(content=user_content)
        ]).content

# Reflection Loop ---
class ReflectionLoop:
    def __init__(self, generator, critic):
        self.generator = generator
        self.critic = critic

    def run(self, question, context, context_map=None, max_retries=0):
        # 1. Start with an initial draft
        current_answer = self.generator.draft(question, context)
        print(f'initial draft: {current_answer}')


        # 2. Iterative loop
        for i in range(max_retries):

            if i==0:
                print('Enter ReflectionLoop')
            feedback = self.critic.verify(current_answer, context)

            print(f'loop 1 feedback: {feedback}')
            if "PASSED" in feedback.upper():
                return current_answer, True

            # 3. Self-Correction step
            current_answer = self.generator.refine_draft(current_answer, feedback)
            print(f'loop 1 ansower: {current_answer}')

        return current_answer, False
