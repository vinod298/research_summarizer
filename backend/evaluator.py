from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def evaluate_answer(question, user_answer, document_text):
    prompt = f"""
Evaluate the following user response.

Question: {question}
User Answer: {user_answer}

Refer to this document content:
{document_text[:2500]}

Give feedback about correctness and justification with paragraph reference if possible.
"""
    return llm.predict(prompt)
