from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gtts import gTTS  # Import gTTS
from io import BytesIO
from fastapi.responses import Response
import pyttsx3


app = FastAPI()
engine=pyttsx3.init()
# Add CORS middleware to allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load PDF and initialize LangChain components
files=["guaranteed-protection-plus-plan-brochure.pdf",
       "indiafirst-life-little-champ-plan-brochure.pdf",
       "gold-brochure.pdf",
       "accidental-death-benefit-rider-brochure.pdf",
       ]

docs=[]
for file_path in files:
    loader = PyPDFLoader(file_path)
    docs.extend(loader.load())
vectorstore = Chroma.from_documents(docs, embedding=GPT4AllEmbeddings())
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=10)
llm = ChatOllama(model="llama3.2")

# Define templates and retrieval logic
message_template = """
Answer this question using the provided context only.
Answer precisely within 3 lines or so,
don't answer with unnecessary answers,
answer only within this PDF,
answer based on the given context shortly,
if the user greets you, greet them back,
always ask a question based on the answer you provided.

{question}

Context:
{context}
"""
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
qa_system_prompt = """
You are an AI insurance assistant, trained to provide precise and professional responses based on the information from the document provided. You are here to assist users with their insurance-related questions using only the context from the document.

### Guidelines for Providing Responses:

- **Respond in 2 to 3 complete sentences** to answer user questions concisely and clearly.
- Use only the information found in the document. If the exact answer is not available, respond with: *"The document does not provide this information."*
- If the user's question is unclear or outside the scope of the document, politely reply with: *"Your question is unclear or falls outside the document's scope. Could you please rephrase or ask a question based on the document's content?"*
- If the user requests an explanation, provide a **brief and simple explanation** in one sentence.
- If the document provides limited information, give the closest matching answer and inform the user about the limitations.
- Always maintain a polite, professional, and helpful tone, keeping your responses clear and accurate.
- If the user greets you, respond politely and professionally before answering their question.
- Make sure your responses are phrased in complete sentences, reflecting the professionalism expected from an insurance agent.

### Context Provided:
{context}                   

### User's Question:
{input}

### Your Response:
"""


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Store chat history in a dictionary
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

class QuestionRequest(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Invoke the RAG chain with the user's question and session ID
        session_history = get_session_history(request.session_id).messages
        response = conversational_rag_chain.invoke(
            {"input": request.question, "chat_history": session_history},
            {"configurable": {"session_id": request.session_id}}  # Pass session_id in config
        )
        # Return the response as JSON
        return JSONResponse(content={"response": response["answer"]})
    except ValueError as e:
        print(f"Error invoking chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
async def home():
    return {"message": "Server is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
