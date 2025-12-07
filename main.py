import os
import logging
import sys
import re
import joblib
import random
import json
import numpy as np
from dotenv import load_dotenv

# --- FastAPI & Pydantic Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- LangChain Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda

# --- NLTK Imports ---
import nltk
from nltk.corpus import stopwords

# --- Custom Engine Import ---
# Ensure insight_engine.py is in the same directory
try:
    from insight_engine import InsightEngine
except ImportError:
    print("⚠ Warning: insight_engine.py not found. Insight generation will fail.")

# --- Setup NLTK ---
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

api_keys_string = os.getenv("GOOGLE_API_KEYS")
if not api_keys_string:
    single_key = os.getenv("GOOGLE_API_KEY")
    if single_key:
        VALID_KEYS = [single_key]
    else:
        raise EnvironmentError("No 'GOOGLE_API_KEYS' (or 'GOOGLE_API_KEY') found in environment variables.")
else:
    VALID_KEYS = [k.strip() for k in api_keys_string.split(",") if k.strip()]

if not VALID_KEYS:
    raise EnvironmentError("API Key list is empty after parsing.")

print(f"✅ Loaded {len(VALID_KEYS)} API Keys.")

def get_random_api_key():
    """Returns a random API key from the pool."""
    return random.choice(VALID_KEYS)

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

class InsightRequest(BaseModel):
    history: List[Dict[str, Any]]  # List of daily metrics from Flutter


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
        google_api_key=VALID_KEYS[0]
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

# --- Load Sentiment Models ---
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

GUIDELINES:
1. Tone: Conversational, gentle, human-like. No academic jargon.
2. Validation: Start by validating their feelings.
3. Context: Use the user's stats ({recent_sentiment}, {avg_sleep} sleep) to personalize advice.
4. Source: Base advice ONLY on the retrieved articles below.
5. Safety: If high risk, ignore this and provide crisis resources.
6. Length: Keep under 100 words.

RETRIEVED KNOWLEDGE:
{context}

CHAT HISTORY:
{chat_history_str}

USER QUERY:
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
# Part 4: Logic Layers (Sentiment & Insights)
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
    
    # Safety Router
    risk_keywords = ["suicide", "kill myself", "want to die", "hurt myself", "end it all"]
    if any(word in user_text for word in risk_keywords):
        return ChatResponse(response="I'm hearing that you're in a lot of pain. Please know that you're not alone. If you are in danger, please call your local emergency number immediately or reach out to a crisis helpline.")

    # Greeting Router
    greetings = ["hi", "hello", "hey", "greetings", "hola"]
    if any(word in user_text for word in greetings) and len(user_text) < 10:
        return ChatResponse(response="Hello! I'm Helm. I'm here to help you navigate your wellness journey. How are you feeling today?")

    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    try:
        current_key = get_random_api_key()
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            temperature=0.7,
            google_api_key=current_key,
            convert_system_message_to_human=True
        )
        
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


# --- NEW: INSIGHT GENERATION ENDPOINT ---
# --- NEW: INSIGHT GENERATION ENDPOINT ---
@app.post("/generate_insights")
async def generate_insights(request: InsightRequest):
    """
    1. Takes user history (List of DailyMetrics).
    2. Runs statistical tests via InsightEngine.
    3. Uses Gemini to narrate the findings into friendly cards.
    4. Passes raw chart data back to frontend.
    """
    try:
        # 1. Run Statistical Analysis
        engine = InsightEngine(request.history)
        raw_findings = engine.run_analysis()
        
        # Handle insufficient data case gracefully
        if isinstance(raw_findings, dict) and "error" in raw_findings:
             return [{
                 "title": "Gathering Data",
                 "summary": "I need a few more days of data to find patterns for you.",
                 "icon": "hourglass_empty",
                 "color": "grey"
             }]
        
        if not raw_findings:
            return [{
                "title": "No Patterns Yet",
                "summary": "Your data is quite balanced right now! Keep tracking to see trends.",
                "icon": "thumb_up",
                "color": "blue"
            }]

        # 2. The "Narrative Layer" (LLM)
        # We need to preserve the chart_data because the LLM might mangle it.
        # Strategy: We ask LLM to generate the text, and we map the chart_data back manually based on index.
        
        current_key = get_random_api_key()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            temperature=0.7,
            google_api_key=current_key
        )

        # Simplify findings for LLM (remove big chart data arrays to save tokens/confusion)
        simplified_findings = []
        for f in raw_findings:
            simple = f.copy()
            if 'chart_data' in simple:
                del simple['chart_data'] # Don't send data points to LLM
            simplified_findings.append(simple)

        prompt = f"""
                You are a friendly data analyst for a mental health app.
                
                *INPUT (Statistical Findings):*
                {json.dumps(simplified_findings, indent=2)}
                
                *TASK:*
                1. Read the statistical findings above.
                2. Generate a user-facing insight card for each item in the list (maintain the order).
                
                *OUTPUT FORMAT (Strict JSON List):*
                [
                {{
                    "title": "Short, catchy title (e.g. 'Sleep is Improving!' or 'Social & Sleep')",
                    "summary": "One clear sentence explaining the pattern simply.",
                    "icon": "Suggest a Flutter Material icon name",
                    "color": "Suggest a color name ('red', 'green', 'orange', 'blue', 'purple')"
                }}
                ]
                
                *RULES:*
                - JSON ONLY. No markdown.
                - If finding is a TREND (type='trend'):
                    - If Metric is Good (Sleep, Mood) & Slope > 0 -> GREEN (Improving).
                    - If Metric is Bad (Stress) & Slope > 0 -> RED (Worsening).
                    - Use icons like 'trending_up' or 'trending_down'.
                - If finding is Correlation/T-Test:
                    - If 'good', use Green/Blue.
                    - If 'warning', use Orange/Red.
                """

        response = await llm.ainvoke(prompt)
        content = response.content.strip()
        
        if content.startswith("json"):
            content = content.replace("json", "").replace("```", "")
            
        insights_json = json.loads(content)
        
        # 3. Re-attach Chart Data
        # We assume LLM returned list in same order as input.
        # We iterate and merge the chart_data back from raw_findings.
        final_insights = []
        for i, insight in enumerate(insights_json):
            if i < len(raw_findings):
                # Copy the LLM generated text
                merged = insight.copy()
                # Attach the raw math/chart data from the engine
                if 'chart_data' in raw_findings[i]:
                    merged['chart_data'] = raw_findings[i]['chart_data']
                final_insights.append(merged)
                
        return final_insights

    except Exception as e:
        print(f"Insight Generation Error: {e}")
        return [{
            "title": "Analyzing Patterns",
            "summary": "We are crunching the numbers. Check back later!",
            "icon": "analytics",
            "color": "blue"
        }]


@app.get("/")
def read_root():
    return {"message": "Helm Wellness API is running."}