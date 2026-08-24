import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "my_collection"
DISTANCE_METRIC = "cosine"
TAVILY_KEY=os.getenv("TAVILY_KEY")

THRESHOLD=0.50