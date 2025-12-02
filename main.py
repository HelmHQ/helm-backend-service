import os
import logging
import sys
import re
import joblib
import random
import numpy as np
from dotenv import load_dotenv

# --- FastAPI & Pydantic Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# --- LangChain Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda

# --- NLTK Imports ---
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# --- Setup Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# --- Load API Keys (Cycling Logic) ---
load_dotenv()

# Load all possible keys
POSSIBLE_KEYS = [
    os.getenv("AIzaSyCH62vARuv43go-hLKQcVD1xkmxO9ePcmw"),
    os.getenv("AIzaSyAiX4DngvqsCSYcCNuSvOfgJcaRaTuyKb0"),
    os.getenv("AIzaSyCMubiEpsgHJYV_GzC12vluEv_yohbP4tQ"),
    os.getenv("AIzaSyBOFE99_ioJOSyGswPRl-_DunE21Ddqit0"),
    os.getenv("AIzaSyCCLSN9DIcXVl21JfRTP4q9lYV3fffpE60"),
    os.getenv("AIzaSyDCLeX4NAnj4rk5mBBCbIyd0jSM1eWb-2k"),
    os.getenv("AIzaSyBjys63rD45gI04GxRpqaMKl6o8oXCd78g"),
    os.getenv("AIzaSyCziUpkpL2CKKURNPVl_62lnQ45RV86cUo"),
    os.getenv("AIzaSyDzZzxgPmaO2bZ8mEHu4BfuDXguKyyUQAc"),
    os.getenv("AIzaSyDH8n4hTOPQtDwREj2xkOzsksvZMmcWrsg")
]
# Filter out None values
VALID_KEYS = [k for k in POSSIBLE_KEYS if k is not None]

if not VALID_KEYS:
    raise EnvironmentError("No GOOGLE_API_KEYs found in environment variables.")

def get_random_api_key():
    """Returns a random API key from the pool."""
    return random.choice(VALID_KEYS)

# Set an initial key for embeddings (embeddings are cheap/fast)
os.environ["GOOGLE_API_KEY"] = VALID_KEYS[0]


# ==========================================
# Part 1: Models & Data Structures
# ==========================================

class HelmContext(BaseModel):
    recent_sentiment: Optional[str] = Field(default="N/A", description="User's most recent journal sentiment")
    screen_time_delta: Optional[str] = Field(default="N/A", description="Change in screen time vs. average")
    avg_sleep: Optional[str] = Field(default="N/A", description="User's recent average sleep")

class ChatHistory(BaseModel):
    role: str = Field(..., description="'user' or 'bot'")
    text: str

class ChatRequest(BaseModel):
    user_query: str
    chat_history: List[ChatHistory]
    helm_context: HelmContext

class ChatResponse(BaseModel):
    response: str


# ==========================================
# Part 2: Initialize App & Load Resources
# ==========================================

app = FastAPI(
    title="Helm Wellness API",
    description="Backend for the Helm digital wellbeing app."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load RAG Vector DB ---
DB_PATH = "./chroma_db"
retriever = None

if os.path.exists(DB_PATH):
    print("Loading RAG components...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_query",
        google_api_key=VALID_KEYS[0] # Use primary key for embeddings
    )
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
else:
    print(f"WARNING: ChromaDB not found at {DB_PATH}. RAG endpoint will fail.")

# --- Load Sentiment Analysis Models ---
vectorizer = None
sentiment_model = None
emotion_names = [
    'afraid', 'angry', 'anxious', 'ashamed', 'awkward', 'bored', 'calm',
    'confused', 'disgusted', 'excited', 'frustrated', 'happy', 'jealous',
    'nostalgic', 'proud', 'sad', 'satisfied', 'surprised'
]

try:
    if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('sentiment_model.pkl'):
        print("Loading Sentiment Models...")
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        sentiment_model = joblib.load('sentiment_model.pkl')
        print("Sentiment Models Loaded.")
    else:
        print("WARNING: .pkl files not found. Sentiment endpoint will fail.")
except Exception as e:
    print(f"Error loading sentiment models: {e}")


# ==========================================
# Part 3: RAG Pipeline Helper Functions
# ==========================================

RAG_PROMPT_TEMPLATE = """
You are 'Helm', a warm, empathetic, and supportive wellness companion. 

*GUIDELINES:*
1. Tone: Conversational, gentle, human-like. No academic jargon.
2. Validation: Start by validating their feelings.
3. Context: Use the user's stats ({recent_sentiment}, {avg_sleep} sleep) to personalize advice.
4. Source: Base advice ONLY on the retrieved articles below.
5. Safety: If high risk, ignore this and provide crisis resources.
6. Length: Keep under 100 words.

*RETRIEVED KNOWLEDGE:*
{context}

*CHAT HISTORY:*
{chat_history_str}

*USER QUERY:*
{question}
"""

def format_docs(docs):
    return "\n\n".join(f"[Article]:\n{doc.page_content}..." for doc in docs)

def format_history(history: List[ChatHistory]):
    if not history: return "No chat history yet."
    return "\n".join(f"{item.role}: {item.text}" for item in history)

def create_refined_query(input_data: dict) -> str:
    context = input_data.get("helm_context")
    query = input_data.get("user_query")
    return f"User sentiment: {context.recent_sentiment}. Sleep: {context.avg_sleep}. Query: {query}"


# ==========================================
# Part 4: Sentiment Analysis Logic
# ==========================================

def cleantext(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'barely', 'hardly'}
    stop_words = stop_words - negation_words
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def predict_emotions_advanced(user_input):
    if not vectorizer or not sentiment_model: return []
    
    cleaned = cleantext(user_input)
    vectorized = vectorizer.transform([cleaned])
    emotions_detected = []
    
    try:
        for i, estimator in enumerate(sentiment_model.estimators_):
            emotion = emotion_names[i]
            threshold = 0.55 if emotion in ['happy', 'calm'] else 0.35
            proba = estimator.predict_proba(vectorized)[0][1]
            
            if proba > threshold:
                emotions_detected.append((emotion, float(proba)))
    except Exception as e:
        print(f"Prediction error: {e}")
        return []
        
    emotions_detected.sort(key=lambda x: x[1], reverse=True)
    return emotions_detected


# ==========================================
# Part 5: API Endpoints
# ==========================================

@app.post("/chat", response_model=ChatResponse)
async def handle_chat_request(request: ChatRequest):
    """RAG Chatbot Endpoint with Pre-Router and Key Cycling"""
    
    user_text = request.user_query.lower()
    
    # --- 1. SAFETY ROUTER (Pre-Check) ---
    risk_keywords = ["suicide", "kill myself", "want to die", "hurt myself", "end it all"]
    if any(word in user_text for word in risk_keywords):
        return ChatResponse(response="I'm hearing that you're in a lot of pain. Please know that you're not alone. If you are in danger, please call your local emergency number immediately or reach out to a crisis helpline.")

    # --- 2. GREETING ROUTER (Pre-Check) ---
    greetings = ["hi", "hello", "hey", "greetings", "hola"]
    # If message is just a greeting (short length)
    if any(word in user_text for word in greetings) and len(user_text) < 10:
        return ChatResponse(response="Hello! I'm Helm. I'm here to help you navigate your wellness journey. How are you feeling today?")

    # --- 3. RAG PIPELINE (Gemini) ---
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    try:
        # Select a fresh API key for this specific request
        current_key = get_random_api_key()
        
        # Initialize LLM with the selected key
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            temperature=0.7,
            google_api_key=current_key, # Dynamic Key Assignment
            convert_system_message_to_human=True
        )
        
        # Re-build chain dynamically (lightweight operation)
        prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        chain = (
            {
                "context": RunnableLambda(create_refined_query) | retriever | RunnableLambda(format_docs),
                "question": lambda x: x["user_query"],
                "chat_history_str": lambda x: format_history(x["chat_history"]),
                "recent_sentiment": lambda x: x["helm_context"].recent_sentiment,
                "avg_sleep": lambda x: x["helm_context"].avg_sleep,
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )

        input_data = {
            "user_query": request.user_query,
            "chat_history": request.chat_history,
            "helm_context": request.helm_context
        }
        
        response_text = await chain.ainvoke(input_data)
        return ChatResponse(response=response_text)

    except Exception as e:
        print(f"RAG Error with key ending in ...{current_key[-4:]}: {e}")
        raise HTTPException(status_code=500, detail="I'm having trouble thinking right now. Please try again.")


@app.post("/analyze_sentiment")
async def analyze_sentiment(request_data: dict):
    """Sentiment Analysis Endpoint"""
    text = request_data.get("text")
    if not text: raise HTTPException(status_code=400, detail="No 'text' provided.")

    if not vectorizer:
        return {"sentiments": ["neutral (offline)"], "scores": [0.0]}

    print(f"Analyzing sentiment for: {text[:30]}...")
    predictions = predict_emotions_advanced(text)

    labels = [e[0] for e in predictions]
    scores = [e[1] for e in predictions]

    if not labels: labels = ["neutral"]
    return {"sentiments": labels, "scores": scores}


@app.get("/")
def read_root():
    return {"message": "Helm Wellness API is running."}