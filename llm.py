import os
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    tokenizer="distilbert-base-uncased-distilled-squad",
    use_auth_token=hf_token
)

def ask_question(context: str, question: str) -> str:
    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"Error: {str(e)}"
