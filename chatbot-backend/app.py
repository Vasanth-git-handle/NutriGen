import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import MongoDBChatMessageHistory
from dotenv import load_dotenv
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# ============ Load environment variables ============
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "diet_chat_db")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")

# ============ Model Config ============
GROQ_MODEL = "llama-3.1-8b-instant"

# ============ Initialize FastAPI ============
app = FastAPI(title="Diet & Nutrition Chatbot (RAG + Persistent Memory)", version="1.1")

# ============ CORS Middleware ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Input schema ============
class Question(BaseModel):
    user_id: Optional[str] = "default_user"
    question: str

# ============ Initialize Groq LLM ============
try:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.0
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

# ============ Load Dataset ============
def load_nutrition(csv_path="Indian_Food_Nutrition_Processed.csv"):
    df = pd.read_csv(csv_path, encoding='utf-8')
    df.columns = df.columns.str.replace('Â', '', regex=False).str.strip()
    df.fillna(0, inplace=True)
    print("🧾 Columns Loaded:", df.columns.tolist())
    return df

nutrition_df = load_nutrition()

# Convert CSV rows to text chunks
nutrition_texts = []
for _, row in nutrition_df.iterrows():
    def safe_get(colnames, default="N/A"):
        for name in colnames:
            if name in row:
                return row[name]
        return default

    text = (
        f"Dish Name: {safe_get(['Dish Name'])}. "
        f"Calories: {safe_get(['Calories (kcal)'])} kcal. "
        f"Carbohydrates: {safe_get(['Carbohydrates (g)'])} g. "
        f"Protein: {safe_get(['Protein (g)'])} g. "
        f"Fats: {safe_get(['Fats (g)', 'Fat (g)'])} g. "
        f"Free Sugar: {safe_get(['Free Sugar (g)'])} g. "
        f"Fibre: {safe_get(['Fibre (g)', 'Fiber (g)'])} g. "
        f"Sodium: {safe_get(['Sodium (mg)'])} mg. "
        f"Calcium: {safe_get(['Calcium (mg)'])} mg. "
        f"Iron: {safe_get(['Iron (mg)'])} mg. "
        f"Vitamin C: {safe_get(['Vitamin C (mg)'])} mg. "
        f"Folate: {safe_get(['Folate (µg)', 'Folate (ug)', 'Folate (mcg)'])} µg."
    )
    nutrition_texts.append(text)

# ============ Create embeddings & FAISS vector store ============
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(nutrition_texts, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ============ Get persistent memory per user ============
def get_user_memory(user_id: str):
    chat_history = MongoDBChatMessageHistory(
        connection_string=MONGO_URI,
        database_name=MONGO_DB,
        collection_name="chat_history",
        session_id=user_id
    )
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=chat_history
    )

# ============ Root endpoint ============
@app.get("/")
def root():
    return {"message": "🥗 Diet & Nutrition Recommendation Chatbot is running successfully!"}

# ============ Ask API ============
@app.post("/ask")
async def ask_diet(question: Question):
    try:
        memory = get_user_memory(question.user_id)

        if "previous question" in question.question.lower():
            history = memory.chat_memory.messages
            user_questions = [m.content for m in history if m.type == "human"]
            if len(user_questions) >= 2:
                return {"answer": f"Your previous question was: {user_questions[-2]}"}
            else:
                return {"answer": "You haven't asked a previous question yet."}

        diet_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a professional diet and nutrition assistant.
Use the following food nutritional data to answer the user's question.
If asked for recommendations, suggest balanced Indian and international meals
based on daily nutrition requirements.
If you don’t find relevant info, reply:
"I'm sorry, I can only answer questions about food and nutrition."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": diet_prompt}
        )

        response = conv_chain.run(question.question)
        return {"answer": response.strip()}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

# ============ Render Deployment Fix (use correct PORT) ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
