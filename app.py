import os
import requests
import pdfplumber
from flask import Flask, json, request, jsonify
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer


load_dotenv()
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
FLASK_PORT = int(os.getenv("FLASK_PORT"))

app = Flask(__name__)

temp_dir = os.path.join(os.path.dirname(__file__), 'temp_files')
os.makedirs(temp_dir, exist_ok=True)

# --- Load embedding model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load ChromaDB ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("resume_guidance")

def load_knowledge_base():
  knowledge_base_path = os.path.join(os.path.dirname(__file__), 'knowledge_base')
  for filename in os.listdir(knowledge_base_path):
    if filename.endswith('.txt'):
      path = os.path.join(knowledge_base_path, filename)
      with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        embedding = embedding_model.encode(content).tolist()
        collection.add(
            documents=[content],
            embeddings=[embedding],
            ids=[filename]
        )
  print(f"Knowledge base loaded with {len(collection.get()['ids'])} entries.")

def retrieve_relevant_context(resume_text, top_k=3):
  resume_embedding = embedding_model.encode(resume_text).tolist()
  results = collection.query(
      query_embeddings=[resume_embedding],
      n_results=top_k
  )
  contexts = results['documents'][0]
  return "\n\n".join(contexts)

def build_ollama_payload(resume_text, rag_context):
  return {
    "model": OLLAMA_MODEL,
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a senior technical recruiter. Review the resume considering best practices "
                "and the following guidelines:\n\n"
                f"{rag_context}\n\n"
                "Provide feedback in exactly this JSON format:"
                '{"strengths": ["strength1", "strength2"], "weaknesses": ["weakness1", "weakness2"],'
                '"suggestions": [{"type": "improvement", "description": "specific actionable advice"}]}'
            )
        },
        {
            "role": "user",
            "content": f"Please review the following resume:\n\n{resume_text}"
        }
    ],
    "stream": False
  }

def call_ollama_api(payload):
  try:
    response = requests.post(
        f"{OLLAMA_API_BASE}/api/chat",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
  except requests.RequestException as e:
    print(f"Error calling OLLAMA API: {e}")
    return None

def extract_text_from_pdf(pdf_path):
  try:
    with pdfplumber.open(pdf_path) as pdf:
      text = ""
      for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text.strip()
  except Exception as e:
    print(f"Error extracting text from PDF: {e}")
    return None

@app.route("/review", methods=["POST"])
def review_resume():
  if 'file' not in request.files:
    return jsonify({"error": "No file part in the request"}), 400
  pdf_file = request.files['file']
  if not pdf_file.filename.endswith('.pdf'):
    return jsonify({"error": "File is not a PDF"}), 400
  
  temp_path = os.path.join(temp_dir, pdf_file.filename)
  print(f"Saving PDF to temporary path: {temp_path}")
  pdf_file.save(temp_path)
  
  resume_text = extract_text_from_pdf(temp_path)
  if not resume_text:
    os.remove(temp_path)
    return jsonify({"error": "Failed to extract text from PDF"}), 500
  print(f"Extracted text from PDF: {resume_text[:250]}...")
  os.remove(temp_path)

  rag_context = retrieve_relevant_context(resume_text)
  payload = build_ollama_payload(resume_text, rag_context)
  ollama_response = call_ollama_api(payload)
  if not ollama_response:
    return jsonify({"error": "Failed to get response from OLLAMA API"}), 500
  return ollama_response

# --- Load KB once ---
load_knowledge_base()

if __name__ == "__main__":
  app.run(port=FLASK_PORT, debug=True)