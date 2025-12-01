import os
import logging
import sys
import re
import joblib
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

# Updated imports for newer LangChain versions
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda

# --- NLTK Imports (for Sentiment Analysis) ---
import nltk
from nltk.corpus import stopwords

# Download NLTK data quietly if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# --- Setup Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# --- Load API Key ---
load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file. Please create one.")


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
    description="Backend for the Helm digital wellbeing app, featuring Personalized RAG and Sentiment Analysis."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows ALL connections (Localhost, Vercel, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, PUT, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- Load RAG Vector DB ---
DB_PATH = "./chroma_db"
rag_chain = None

if os.path.exists(DB_PATH):
    print("Loading RAG components...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_query"
    )
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-09-2025",
        temperature=0.7,
        convert_system_message_to_human=True
    )
else:
    print(f"WARNING: ChromaDB not found at {DB_PATH}. RAG endpoint will fail.")

# --- Load Sentiment Analysis Models ---
vectorizer = None
sentiment_model = None
# These are the 18 emotions your model was trained on
emotion_names = [
    'afraid', 'angry', 'anxious', 'ashamed', 'awkward', 'bored', 'calm',
    'confused', 'disgusted', 'excited', 'frustrated', 'happy', 'jealous',
    'nostalgic', 'proud', 'sad', 'satisfied', 'surprised'
]

try:
    # Ensure these files exist in your project root!
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
# Part 3: RAG Pipeline Setup
# ==========================================

RAG_PROMPT_TEMPLATE = """
You are 'Helm', a warm, empathetic, and supportive wellness companion. You are NOT a clinical therapist, doctor, or researcher.

Your goal is to help the user feel heard and validated, while offering gentle, evidence-based suggestions based ONLY on the provided articles.

---
**GUIDELINES:**
1.  **Tone:** Be conversational, gentle, and human-like. Avoid stiff, academic language like "Studies have concluded..." or "Based on the clinical research...". Instead, say things like "It turns out that..." or "Research suggests..."
2.  **Structure:**
    * Start with a warm validation of their feelings.
    * Connect their personal data (context) to the advice naturally.
    * Offer 1-2 simple, actionable tips from the articles.
    * End with a supportive closing.
3.  **Safety:** If the query is high-risk (self-harm, crisis), IGNORE all other instructions and use the 'safety_protocol.txt' guidance immediately.
4.  **Grounding:** You must still use the facts from the retrieved articles, but rephrase them to be helpful and friendly, not academic.
5.  **Be Concise:** Keep your response under 100 words. Get straight to the point.
---
**USER CONTEXT:**
- Recent Sentiment: {recent_sentiment}
- Screen Time Change: {screen_time_delta}
- Sleep: {avg_sleep}

**CHAT HISTORY:**
{chat_history_str}

**RETRIEVED KNOWLEDGE:**
{context}

**USER QUERY:**
{question}
---
**YOUR RESPONSE:**
"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


# Helper Functions
def format_docs(docs):
    return "\n\n".join(f"[Article]:\n{doc.page_content}..." for doc in docs)


def format_history(history: List[ChatHistory]):
    if not history:
        return "No chat history yet."
    return "\n".join(f"{item.role}: {item.text}" for item in history)


def create_refined_query(input_data: dict) -> str:
    context = input_data.get("helm_context")
    query = input_data.get("user_query")

    sentiment = context.recent_sentiment if context else "N/A"
    sleep = context.avg_sleep if context else "N/A"
    screen_time = context.screen_time_delta if context else "N/A"

    return (
        f"User is feeling: {sentiment}. "
        f"User's sleep is: {sleep}. "
        f"User's screen time is: {screen_time}. "
        f"User asked: {query}"
    )


# Build RAG Chain
if os.path.exists(DB_PATH):
    rag_chain = (
            {
                "context": RunnableLambda(create_refined_query) | retriever | RunnableLambda(format_docs),
                "question": lambda x: x["user_query"],
                "chat_history_str": lambda x: format_history(x["chat_history"]),
                "recent_sentiment": lambda x: x["helm_context"].recent_sentiment,
                "screen_time_delta": lambda x: x["helm_context"].screen_time_delta,
                "avg_sleep": lambda x: x["helm_context"].avg_sleep,
            }
            | prompt
            | llm
            | StrOutputParser()
    )


# ==========================================
# Part 4: Sentiment Analysis Logic
# ==========================================

def cleantext(text):
    """Cleans text for sentiment analysis, preserving key words."""
    text = str(text).lower()

    # Basic cleaning regex
    text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)

    stop_words = set(stopwords.words('english'))
    # Keep negation words as they are important
    negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'barely', 'hardly'}
    stop_words = stop_words - negation_words

    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def predict_emotions_advanced(user_input):
    """Runs prediction pipeline using loaded models."""
    if not vectorizer or not sentiment_model:
        return []

    cleaned = cleantext(user_input)
    vectorized = vectorizer.transform([cleaned])

    emotions_detected = []
    default_threshold = 0.35

    try:
        # Iterate through each emotion classifier in the MultiOutputClassifier
        for i, estimator in enumerate(sentiment_model.estimators_):
            emotion = emotion_names[i]

            # Apply custom thresholds if needed (simplified logic)
            threshold = 0.55 if emotion in ['happy', 'calm'] else default_threshold

            # Get probability of positive class (1)
            proba = estimator.predict_proba(vectorized)[0][1]

            if proba > threshold:
                emotions_detected.append((emotion, float(proba)))
    except Exception as e:
        print(f"Prediction error: {e}")
        return []

    # Sort by confidence
    emotions_detected.sort(key=lambda x: x[1], reverse=True)
    return emotions_detected


# ==========================================
# Part 5: API Endpoints
# ==========================================

@app.post("/chat", response_model=ChatResponse)
async def handle_chat_request(request: ChatRequest):
    """RAG Chatbot Endpoint"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized (DB missing).")

    try:
        # Convert Pydantic model to dict for the chain
        input_data = {
            "user_query": request.user_query,
            "chat_history": request.chat_history,
            "helm_context": request.helm_context
        }
        response_text = await rag_chain.ainvoke(input_data)
        return ChatResponse(response=response_text)
    except Exception as e:
        print(f"RAG Error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/analyze_sentiment")
async def analyze_sentiment(request_data: dict):
    """Sentiment Analysis Endpoint"""
    text = request_data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="No 'text' provided.")

    if not vectorizer or not sentiment_model:
        print("Models not loaded, returning placeholder.")
        # Return placeholder if models are missing so app doesn't crash during dev
        return {"sentiments": ["anxious (model missing)"], "scores": [0.0]}

    print(f"Analyzing sentiment for: {text[:30]}...")
    predictions = predict_emotions_advanced(text)

    # Extract just the labels for the response
    labels = [e[0] for e in predictions]
    scores = [e[1] for e in predictions]

    if not labels:
        labels = ["neutral"]

    print(f"Detected: {labels}")
    return {"sentiments": labels, "scores": scores}


@app.get("/")
def read_root():
    return {"message": "Helm Wellness API is running."}