import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. Configure Google Gemini with Auto-Discovery
active_model = None

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # --- NEW: AUTO-DETECT WORKING MODEL ---
    def get_working_model():
        try:
            # List all models available to your specific API Key
            print("🔄 Checking available models...")
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            # Priority list (try these first)
            priorities = ['models/gemini-1.5-flash', 'models/gemini-pro', 'models/gemini-1.0-pro']
            
            # Pick the first priority that exists in your available list
            for p in priorities:
                if p in available_models:
                    print(f"✅ Selected Model: {p}")
                    return genai.GenerativeModel(p)
            
            # Fallback: Just pick the first available one
            if available_models:
                print(f"⚠️ Fallback Model: {available_models[0]}")
                return genai.GenerativeModel(available_models[0])
                
        except Exception as e:
            print(f"❌ Error listing models: {e}")
        return None

    active_model = get_working_model()

# 3. Embedding Function (Matches your data_processor.py)
class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, input):
        if isinstance(input, str): input = [input]
        return self.model.encode(input).tolist()

# 4. Database Connection
@st.cache_resource
def get_db():
    try:
        # POINT TO THE EXACT FOLDER WHERE YOU RAN DATA_PROCESSOR
        client = chromadb.PersistentClient(path="./chroma_db")
        return client.get_collection("arxiv_papers", embedding_function=LocalEmbeddingFunction())
    except Exception as e:
        print(f"DB Error: {e}")
        return None

# ==========================================
# UI LAYOUT
# ==========================================
st.set_page_config(page_title="ArXiv Search", layout="wide")
st.title("🚀 ArXiv Intelligent Search")

# Sidebar
with st.sidebar:
    st.header("Status")
    if active_model:
        st.success(f"🟢 AI Model Connected")
    else:
        st.error("🔴 AI Model Failed (Check Terminal)")
        
    collection = get_db()
    if collection:
        st.info(f"📚 {collection.count()} Papers Indexed")
    else:
        st.warning("⚠️ Database not found")

# Main Search
query = st.text_input("Enter research query:", "What are the limitations of Transformer models?")
col1, col2 = st.columns(2)
with col1:
    n_results = st.slider("Papers to analyze", 3, 10, 5)
with col2:
    enable_llm = st.checkbox("Enable AI Synthesis", value=True)

if st.button("Search", type="primary"):
    if not collection:
        st.error("Database missing. Please run data_processor.py first.")
    else:
        with st.spinner("Analyzing papers..."):
            # 1. Search Vector DB
            results = collection.query(query_texts=[query], n_results=n_results)
            
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            # 2. Display Papers
            st.subheader("📄 Relevant Papers")
            context_text = ""
            
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                with st.expander(f"{i+1}. {meta.get('title', 'Untitled')} ({meta.get('year', 'N/A')})"):
                    st.markdown(f"**Abstract:** {doc}")
                
                # Add to context for AI
                context_text += f"Paper {i+1}: {meta.get('title')}\nAbstract: {doc}\n\n"

            # 3. AI Synthesis
            if enable_llm and active_model:
                st.markdown("---")
                st.subheader("🤖 AI Insights")
                try:
                    prompt = f"""
                    You are a researcher. Answer the following question based ONLY on the papers provided below.
                    
                    QUESTION: {query}
                    
                    PAPERS:
                    {context_text}
                    
                    INSTRUCTIONS:
                    - Synthesize the answer.
                    - Cite papers by number (e.g. [Paper 1]).
                    - Be concise.
                    """
                    
                    response = active_model.generate_content(prompt)
                    st.info(response.text)
                    
                except Exception as e:
                    st.error(f"AI Generation Failed: {e}")
            elif enable_llm and not active_model:
                 st.warning("AI features disabled because no valid Google Model was found.")