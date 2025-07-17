import os
import requests
import pdfplumber
from flask import Flask, json, request, jsonify
from dotenv import load_dotenv

load_dotenv()
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
FLASK_PORT = int(os.getenv("FLASK_PORT"))

temp_dir = os.path.join(os.path.dirname(__file__), 'temp_files')
os.makedirs(temp_dir, exist_ok=True)

app = Flask(__name__)

def build_ollama_payload(resume_text):
  return {
    "model": OLLAMA_MODEL,
    "messages": [
        {
            "role": "system",
            "content": (
              "You are a professional technical resume reviewer consulted by the top tech firms for shortlisting candidates. "
              "Given a candidate's resume, provide constructive feedback on their strengths, weaknesses, and suggestions for improvement."
              "The feedback should be detailed and actionable, focusing on both technical skills and overall presentation. Keep the feedback concise and professional."
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
    return response.json()
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

  payload = build_ollama_payload(resume_text)
  ollama_response = call_ollama_api(payload)
  if not ollama_response:
    return jsonify({"error": "Failed to get response from OLLAMA API"}), 500
  return ollama_response

if __name__ == "__main__":
  app.run(port=FLASK_PORT, debug=True)