from fastapi import FastAPI
from pydantic import BaseModel
from app.utils.llm import ask_question

app = FastAPI()

class QueryInput(BaseModel):
    paper: str
    query: str

@app.post("/query")
async def query_paper(input: QueryInput):
    paper_text = input.paper
    question = input.query

    answer = ask_question(paper_text, question)
    return {"answer": answer}
