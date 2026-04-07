# LCEL
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.agent import prompts
#import re

# The Generator (Drafting & Refining)
class ResearchGenerator:
    def __init__(self, llm):
        # Initial Drafting Chain
        self.draft_chain = (
            ChatPromptTemplate.from_messages([
                ("system", prompts.GENERATOR_SYSTEM),
                ("human", "Context:\n{context}\n\nQuestion: {question}")
            ])
            | llm | StrOutputParser()
        )

        # Refining Chain
        self.refine_chain = (
            ChatPromptTemplate.from_messages([
                ("system", prompts.REFINER_SYSTEM),
                ("human", "Original Draft: {draft}\n\nFeedback: {feedback}\n\nRewrite the answer correctly.")
            ])
            | llm | StrOutputParser()
        )

# The Critic (Verification)
class ResearchCritic:
    def __init__(self, llm):
        self.verify_chain = (
            ChatPromptTemplate.from_messages([
                ("system", prompts.CRITIC_SYSTEM),
                ("human", prompts.CRITIC_USER_TEMPLATE)
            ])
            | llm | StrOutputParser()
        )
