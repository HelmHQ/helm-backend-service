import os
import logging
import sys
import re
import joblib
import random
import json
import traceback # --- NEW: For better error logging
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
from langchain_core.documents import Document 

# --- NLTK Imports ---
import nltk
from nltk.corpus import stopwords

# --- Custom Engine Import ---
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

# --- Load API Keys ---
load_dotenv()
api_keys_string = os.getenv("GOOGLE_API_KEYS")
if not api_keys_string:
    single_key = os.getenv("GOOGLE_API_KEY")
    if single_key: VALID_KEYS = [single_key]
    else: raise EnvironmentError("No GOOGLE_API_KEYS found.")
else:
    VALID_KEYS = [k.strip() for k in api_keys_string.split(",") if k.strip()]

if not VALID_KEYS: raise EnvironmentError("API Key list is empty.")

def get_random_api_key():
    return random.choice(VALID_KEYS)

os.environ["GOOGLE_API_KEY"] = VALID_KEYS[0]

# ==========================================
# Part 1: Models
# ==========================================

class HelmContext(BaseModel):
    recent_sentiment: Optional[str] = Field(default="N/A")
    screen_time_delta: Optional[str] = Field(default="N/A")
    avg_sleep: Optional[str] = Field(default="N/A")

class ChatHistory(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    user_query: str
    chat_history: List[ChatHistory]
    helm_context: HelmContext

class ChatResponse(BaseModel):
    response: str

class InsightRequest(BaseModel):
    history: List[Dict[str, Any]] 

# ==========================================
# Part 2: App Setup & Routing Initialization
# ==========================================

app = FastAPI(title="Helm Wellness API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- A. Load RAG Knowledge Base ---
DB_PATH = "./chroma_db"
retriever = None
if os.path.exists(DB_PATH):
    print("Loading RAG components...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_query", google_api_key=VALID_KEYS[0])
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# --- B. Load Sentiment Models ---
vectorizer = None
sentiment_model = None
emotion_names = ['afraid', 'angry', 'anxious', 'ashamed', 'awkward', 'bored', 'calm', 'confused', 'disgusted', 'excited', 'frustrated', 'happy', 'jealous', 'nostalgic', 'proud', 'sad', 'satisfied', 'surprised']

try:
    if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('sentiment_model.pkl'):
        print("Loading Sentiment Models...")
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        sentiment_model = joblib.load('sentiment_model.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

# --- C. Initialize Semantic Router ---
print("Initializing Semantic Router...")
router_store = None
try:
    # 1. Define Anchors
    anchors = [
        # Crisis Anchors (High Priority)
        Document(page_content="I want to kill myself", metadata={"intent": "crisis"}),
        Document(page_content="I want to die", metadata={"intent": "crisis"}),
        Document(page_content="I am suicidal", metadata={"intent": "crisis"}),
        Document(page_content="I'm going to end it all", metadata={"intent": "crisis"}),
        Document(page_content="I want to hurt myself", metadata={"intent": "crisis"}),
        
        # Greeting Anchors
        Document(page_content="Hello", metadata={"intent": "greeting"}),
        Document(page_content="Hi there", metadata={"intent": "greeting"}),
        Document(page_content="Good morning", metadata={"intent": "greeting"}),
        Document(page_content="Hey Helm", metadata={"intent": "greeting"}),
        
        # Gratitude Anchors
        Document(page_content="Thank you so much", metadata={"intent": "gratitude"}),
        Document(page_content="That was helpful", metadata={"intent": "gratitude"}),
        Document(page_content="Thanks", metadata={"intent": "gratitude"}),
    ]
    
    # 2. Create in-memory vector store for routing
    router_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_query", google_api_key=VALID_KEYS[0])
    router_store = Chroma.from_documents(anchors, router_embeddings, collection_name="router_anchors")
    print("✅ Semantic Router Ready")
except Exception as e:
    print(f"❌ Router Initialization Failed: {e}")

# ==========================================
# Part 3: Helpers
# ==========================================

RAG_PROMPT = """You are Helm, a warm, empathetic, and supportive wellness companion.
GUIDELINES: Warm, conversational, validate feelings. Use user context. Base advice ONLY on retrieved articles.
CONTEXT: {context}
HISTORY: {chat_history_str}
QUERY: {question}"""

def format_docs(docs): return "\n\n".join(f"[Article]:\n{doc.page_content}..." for doc in docs)
def format_history(history): return "\n".join(f"{item.role}: {item.text}" for item in history) if history else "No history."
def create_refined_query(input_data):
    ctx = input_data.get("helm_context")
    return f"Sentiment: {ctx.recent_sentiment}. Sleep: {ctx.avg_sleep}. Query: {input_data.get('user_query')}"

def cleantext(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
    return text

def predict_emotions(user_input):
    if not vectorizer or not sentiment_model: return []
    cleaned = cleantext(user_input)
    vectorized = vectorizer.transform([cleaned])
    emotions = []
    for i, estimator in enumerate(sentiment_model.estimators_):
        proba = estimator.predict_proba(vectorized)[0][1]
        if proba > (0.55 if emotion_names[i] in ['happy', 'calm'] else 0.35):
            emotions.append((emotion_names[i], float(proba)))
    emotions.sort(key=lambda x: x[1], reverse=True)
    return emotions

# ==========================================
# Part 4: Endpoints
# ==========================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Semantic Routing + RAG Chatbot"""
    
    # --- STEP 1: SEMANTIC ROUTING ---
    if router_store:
        try:
            # Search for similar anchor
            results = router_store.similarity_search_with_score(request.user_query, k=1)
            if results:
                doc, score = results[0]
                # Chroma returns 'distance' (lower is better).
                # Typically 0.0 is exact match. > 0.4 is getting irrelevant.
                
                if score < 0.35:
                    intent = doc.metadata["intent"]
                    
                    if intent == "crisis":
                        return ChatResponse(response="I'm hearing that you're in a lot of pain. Please know that you're not alone. If you are in danger, please call your local emergency number immediately or reach out to a crisis helpline.")
                    
                    if intent == "greeting":
                        return ChatResponse(response="Hello! I'm Helm. I'm here to help you navigate your wellness journey. How are you feeling today?")
                        
                    if intent == "gratitude":
                        return ChatResponse(response="You're very welcome! I'm glad I could help. Is there anything else on your mind?")
        except Exception as e:
            print(f"Router Error: {e}")
            # Continue to RAG if router fails

    # --- STEP 2: RAG PIPELINE ---
    if not retriever: raise HTTPException(503, "RAG not initialized")
    
    try:
        current_key = get_random_api_key()
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0.7, google_api_key=current_key)
        chain = (
            {"context": RunnableLambda(create_refined_query) | retriever | RunnableLambda(format_docs),
             "question": lambda x: x["user_query"],
             "chat_history_str": lambda x: format_history(x["chat_history"]),
             "recent_sentiment": lambda x: x["helm_context"].recent_sentiment,
             "avg_sleep": lambda x: x["helm_context"].avg_sleep}
            | ChatPromptTemplate.from_template(RAG_PROMPT)
            | llm
            | StrOutputParser()
        )
        # --- FIX: Use model_dump() instead of dict() for Pydantic v2 compatibility ---
        return ChatResponse(response=await chain.ainvoke(request.model_dump()))
    except Exception as e:
        # --- FIX: Print full traceback for debugging ---
        traceback.print_exc()
        raise HTTPException(500, f"Error: {e}")

@app.post("/analyze_sentiment")
async def analyze(request: dict):
    text = request.get("text")
    if not text: raise HTTPException(400, "No text")
    preds = predict_emotions(text)
    return {"sentiments": [p[0] for p in preds] or ["neutral"], "scores": [p[1] for p in preds] or [0.0]}

@app.post("/generate_insights")
async def generate_insights(request: InsightRequest):
    try:
        engine = InsightEngine(request.history)
        raw_findings = engine.run_analysis()

        if isinstance(raw_findings, dict) and "error" in raw_findings:
            return [{
                "title": "Gathering Data",
                "summary": "I need about 5 days of data to start finding personal patterns for you.",
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

        current_key = get_random_api_key()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            temperature=0.7,
            google_api_key=current_key
        )

        simplified_findings = []
        for f in raw_findings:
            simple = f.copy()
            if 'chart_data' in simple:
                del simple['chart_data'] 
            simplified_findings.append(simple)

        prompt = f"""
        You are a friendly data analyst for a wellness app.
        
        *INPUT (Statistical Findings):*
        {json.dumps(simplified_findings, indent=2)}
        
        *TASK:*
        1. Read the statistical findings above.
        2. Select the top 3 most significant/interesting ones.
        3. Translate them into friendly, non-technical insight cards.
        
        *OUTPUT FORMAT (Strict JSON List):*
        [
          {{
            "title": "Short Title (e.g. 'Social & Sleep')",
            "summary": "One clear sentence explaining the finding.",
            "icon": "Flutter Icon Name",
            "color": "Color Name"
          }}
        ]
        
        *RULES:*
        - JSON ONLY. No markdown.
        - If finding is a TREND (type='trend'):
            - Metric Good & Slope > 0 -> GREEN (Improving).
            - Metric Bad & Slope > 0 -> RED (Worsening).
            - Use icons 'trending_up', 'trending_down'.
        - If finding is Correlation/T-Test:
            - Good -> Green/Blue.
            - Warning -> Orange/Red.
        """

        response = await llm.ainvoke(prompt)
        content = response.content.strip()
        
        if content.startswith("json"):
            content = content.replace("json", "").replace("```", "")
            
        insights_json = json.loads(content)
        
        final_insights = []
        for i, insight in enumerate(insights_json):
            if i < len(raw_findings):
                merged = insight.copy()
                if 'chart_data' in raw_findings[i]:
                    merged['chart_data'] = raw_findings[i]['chart_data']
                final_insights.append(merged)
                
        return final_insights

    except Exception as e:
        print(f"Insight Error: {e}")
        traceback.print_exc()
        return [{
            "title": "Analyzing...",
            "summary": "We are crunching the numbers. Check back later!",
            "icon": "analytics",
            "color": "blue"
        }]

@app.get("/")
def root(): return {"message": "Helm API Running"}