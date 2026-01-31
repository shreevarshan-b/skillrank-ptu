import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode([str(x) for x in input]).tolist()

class AdvancedDataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="arxiv_papers",
            embedding_function=LocalEmbeddingFunction()
        )

    def load_papers(self, limit: int = 1000) -> List[Dict]:
        print(f"📂 Reading file: {self.data_path}")
        papers = []
        
        try:
            # METHOD 1: Try loading as a standard JSON List (Most likely for your file)
            with open(self.data_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == '[':
                    print("✅ Detected JSON List format. Loading entire file...")
                    raw_data = json.load(f)
                    print(f"✅ Loaded {len(raw_data)} raw records. Processing...")
                    
                    for i, obj in enumerate(raw_data):
                        if len(papers) >= limit: break
                        p = self._process_single_paper(obj, i)
                        if p: papers.append(p)
                    return papers
                
                # METHOD 2: JSON Lines (One object per line)
                else:
                    print("✅ Detected JSON Lines format. Reading line by line...")
                    for i, line in enumerate(f):
                        if len(papers) >= limit: break
                        try:
                            obj = json.loads(line)
                            p = self._process_single_paper(obj, i)
                            if p: papers.append(p)
                        except: continue
                        
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            
        return papers

    def _process_single_paper(self, obj: Dict, index: int) -> Dict:
        try:
            # 1. robust Abstract Extraction
            abstract = "No Abstract"
            # List of possible keys for abstract
            keys_to_check = ['abstract', 'Abstract', 'summary', 'Summary', 'description', 'Description']
            for k in keys_to_check:
                if k in obj and obj[k]:
                    abstract = str(obj[k]).strip()
                    break
            
            # 2. robust Title Extraction
            title = obj.get('title', obj.get('Title', 'No Title'))
            title = str(title).strip().replace('\n', ' ')
            
            # 3. robust Year Extraction
            year = 'N/A'
            if 'versions' in obj and isinstance(obj['versions'], list) and obj['versions']:
                year = obj['versions'][0].get('created', '')[:4]
            elif 'year' in obj:
                year = str(obj['year'])
            elif 'date' in obj:
                year = str(obj['date'])[:4]

            paper_id = str(obj.get('id', index))

            return {
                "id": paper_id,
                "text": f"Title: {title}\nAbstract: {abstract}",
                "metadata": {
                    "paper_id": paper_id,
                    "title": title[:100],
                    "year": year
                }
            }
        except Exception as e:
            return None

    def build_index(self, papers: List[Dict]):
        if not papers: return
        print(f"🚀 Indexing {len(papers)} papers...")
        
        batch_size = 50
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            try:
                self.collection.upsert(
                    ids=[p['id'] for p in batch],
                    documents=[p['text'] for p in batch],
                    metadatas=[p['metadata'] for p in batch]
                )
                if i % 100 == 0: print(f"   Indexed batch {i}")
            except Exception as e:
                print(f"⚠️ Batch Error: {e}")

if __name__ == "__main__":
    import os
    import shutil
    
    # 👇 KEEP YOUR PATH
    MY_DATA_PATH = r"D:\skillrank\arxiv-search\data\arxiv-metadata-oai-snapshot.json" 
    
    if os.path.exists(MY_DATA_PATH):
        # 1. Force Delete Old DB to fix "No Abstract"
        if os.path.exists("./chroma_db"):
            try:
                shutil.rmtree("./chroma_db")
                print("🗑️ Database reset (Necessary for new abstracts).")
            except: pass

        # 2. Run
        processor = AdvancedDataProcessor(MY_DATA_PATH)
        papers = processor.load_papers(limit=2000)
        
        if papers:
            processor.build_index(papers)
            print("🎉 DONE! Data is fixed. Now run: python -m streamlit run final_app.py")
        else:
            print("❌ No papers found. The file might be empty.")
    else:
        print("❌ File not found.")