import os
import numpy as np
from dotenv import load_dotenv
import time
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_astradb import AstraDBVectorStore
from embedded_questions import embedded_questions

# Functions Below

def get_top_relevant_questions(ai_response, embedded_questions, embedding_model, top_k=3):
    """
    Converts the ai_response into an embedding, compares it with all embedded_questions,
    and returns the top K most relevant questions (default: 3).
    """
    # Get embedding for AI response
    response_embedding = embedding_model.embed_query(ai_response)

    # Compute cosine similarity between response and each stored question
    similarities = []
    for item in embedded_questions:
        question_embedding = np.array(item["embedding"])
        similarity = np.dot(response_embedding, question_embedding) / (
            np.linalg.norm(response_embedding) * np.linalg.norm(question_embedding)
        )
        similarities.append((item["question"], similarity))

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top K question texts
    top_questions = [q for q, _ in similarities[:top_k]]

    return top_questions

# Functions Above

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
)

# Reload vector DB (no re-embedding, fast)
vector_store = AstraDBVectorStore(
        collection_name="Madvisions_Data",       
        embedding=embedding_model,
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],       
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],           
        namespace=None         
)

contextualize_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

system_prompt = """
You are the Madvisions assistant chatbot, helping users with questions about Madvisions and its services with creativity, clarity, and confidence. 
Always respond based on the provided context and focus only on what Madvisions offers — explain, recommend, or guide users toward relevant Madvisions services, not general ideas or advice. 
Keep your responses clear, friendly, and professional — no longer than 3 to 4 lines. Be concise but complete, ensuring the user understands how Madvisions can help. 
After completing your main response, insert two line breaks, and then add a short follow-up sentence such as: "Would you like me to explain this in more detail?" or "Would you like to know more about this service?".
If the question is outside Madvisions services or unrelated to what Madvisions provides, politely respond: 
"I am here to assist with Madvisions services only."
Do not provide unrelated or speculative ideas.
Context: {context}
"""

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

retriever = vector_store.as_retriever(search_kwargs={'k': 3})
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt,
)
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    document_chain
)

chat_histories: Dict[str, List] = {}
session_timestamps = {}

# Routes

class UserInput(BaseModel):
    user_input: str
    session_id: str

@app.post("/ai-answer")
def generate_answer(request: UserInput):

    # Clean expired sessions (10 min)
    for sid in list(session_timestamps.keys()):
        if time.time() - session_timestamps[sid] > 600:
            chat_histories.pop(sid, None)
            session_timestamps.pop(sid, None)
    
    session_id = request.session_id

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    session_timestamps[session_id] = time.time()

    try:
        # Get RAG-based AI answer
        response = rag_chain.invoke({
            "chat_history": chat_histories[session_id],
            "input": request.user_input,
        })

        ai_answer = response["answer"]

        # Add conversation to chat history
        chat_histories[session_id].extend([
            HumanMessage(content=request.user_input),
            AIMessage(content=ai_answer)
        ])

        # Get top 3 relevant questions
        top_questions = get_top_relevant_questions(
            ai_response=ai_answer,
            embedded_questions=embedded_questions,
            embedding_model=embedding_model,
            top_k=3
        )

        return {
            "answer": ai_answer,
            "related_questions": top_questions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Below is just for testing and running server locally (remove below when uploading it on a cloud)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
