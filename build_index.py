import os
import shutil
import logging
import sys
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Setup Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# --- Load API Key ---
load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file.")

print("--- Phase 2: Building Vector Database (Robust Mode) ---")

KNOWLEDGE_BASE_DIR = "./knowledge_base"
DB_PATH = "./chroma_db"


def build_vector_database():
    # --- 1. Load Documents (Robust Manual Loop) ---
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Error: {KNOWLEDGE_BASE_DIR} not found.")
        return

    print(f"Scanning files in {KNOWLEDGE_BASE_DIR}...")

    documents = []

    # Walk through the directory manually
    for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    # Load PDF
                    print(f"Loading PDF: {file}...")
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())

                elif file.endswith((".txt", ".md")):
                    # Load Text/Markdown with explicit UTF-8 encoding
                    print(f"Loading Text: {file}...")
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())

            except Exception as e:
                print(f"⚠️ Failed to load {file}: {e}")
                continue

    if not documents:
        print("No documents were loaded. Exiting.")
        return

    print(f"\n✅ Successfully loaded {len(documents)} total document pages/files.")

    # --- 2. Chunk Documents ---
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        print("No chunks created. Exiting.")
        return

    print(f"Created {len(chunks)} searchable chunks.")

    # --- 3. Initialize Embedding Model ---
    print("Initializing Gemini embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_document",
        title="Helm Knowledge Base Document"
    )

    # --- 4. Embed and Persist ---
    if os.path.exists(DB_PATH):
        print(f"Cleaning up old database at {DB_PATH}...")
        try:
            shutil.rmtree(DB_PATH)
        except Exception as e:
            print(f"Warning: Could not fully delete old DB ({e}). Continuing...")

    print(f"Building database at {DB_PATH}. This may take a few minutes...")

    try:
        # Create and persist the database
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        print(f"\n✅ Success! Vector Database built at '{DB_PATH}'")

    except Exception as e:
        print(f"\n❌ Critical Error during embedding: {e}")


if __name__ == "__main__":
    build_vector_database()