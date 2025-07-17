# Resume Reviewer Agent

A Flask-based application for automated resume review using LLMs. This project extracts text from uploaded PDF resumes and provides structured feedback on strengths, weaknesses, and suggestions.

## Getting Started

### Prerequisites

- Python 3.8+
- Docker

## Hosting Ollama with Docker

Ollama is used to run the LLM models locally.

1. **Start the Ollama Docker container:**
   ```bash
   docker run -d --name ollama \
     -p 11434:11434 \
     -v ollama_data:/root/.ollama \
     ollama/ollama
   ```
   Ollama will now be accessible at `http://localhost:11434`

2. **Pull the `phi3` model:**
   From running docker container execute the following command.
   ```bash
   ollama pull phi3
   ```
   Once the `phi3` model is pulled, you can test it with the following cURL command:

   ```bash
   curl http://localhost:11434/api/chat -d '{
   "model": "phi3",
   "messages": [{"role": "user", "content": "Hello"}]
   }'
   ```
   This will send a test message to the Phi3 model running in your Ollama Docker container.

### Flask App Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SourabhJaz/resume-reviewer-agent.git
   cd resume-reviewer-agent
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.lock.txt
   ```

### Running the Flask App

```bash
python app.py
```

## Example: Submitting a Resume for Review via cURL

You can submit a PDF resume to the API using the following cURL command:

```bash
curl --location 'http://127.0.0.1:5005/review' \
  --form 'file=@"/Users/usenamw/Documents/resume.pdf"'
```

Replace the file path with the path to your own
