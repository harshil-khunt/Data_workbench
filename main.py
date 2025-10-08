# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import pandas as pd
import io
import plotly.express as px
import uuid
import os
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    ai_model = genai.GenerativeModel('models/gemini-2.5-flash')
except Exception as e:
    print(f"Error configuring Google AI: {e}")
    ai_model = None

app = FastAPI()
df_cache = {}

class ChatQuestion(BaseModel):
    question: str

# ==============================================================================
# 3. FRONTEND - The Main Hub and Spoke Pages
# ==============================================================================
@app.get("/")
def read_root():
    """Serves the main landing page for file upload."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="utf-8"><title>AI Data Workbench</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="bg-light">
            <div class="container mt-5">
                <div class="card shadow-sm">
                    <div class="card-body">
                        <h1 class="card-title mb-4">AI Data Workbench</h1>
                        <p class="card-text">Upload a CSV file to begin cleaning, analyzing, and chatting with your data.</p>
                        <form action="/upload/" enctype="multipart/form-data" method="post" class="mt-4">
                            <div class="mb-3">
                                <input name="file" type="file" class="form-control" accept=".csv" required>
                            </div>
                            <button type="submit" class="btn btn-primary">Process File</button>
                        </form>
                    </div>
                </div>
            </div>
        </body>
    </html>
    """)

@app.get("/dashboard/{file_id}")
def get_dashboard(file_id: str):
    """Serves the central dashboard after a file is uploaded."""
    if file_id not in df_cache:
        return RedirectResponse(url="/")
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="utf-8"><title>Data Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="text-center">Your Data is Ready</h1>
                <p class="text-center text-muted">Choose an action below to analyze your dataset.</p>
                <div class="row mt-4 g-3">
                    <div class="col-md-4">
                        <div class="card h-100">
                            <div class="card-body text-center">
                                <h5 class="card-title">🤖 AI Visualization Report</h5>
                                <p class="card-text">Let our AI automatically generate the most insightful charts from your data.</p>
                                <a href="/ai_visuals/{file_id}" class="btn btn-primary" target="_blank">Generate AI Visuals</a>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card h-100">
                            <div class="card-body text-center">
                                <h5 class="card-title">📊 Detailed Statistical Profile</h5>
                                <p class="card-text">Get a deep-dive statistical report covering every column in your dataset.</p>
                                <a href="/statistical_report/{file_id}" class="btn btn-secondary" target="_blank">Generate Stat Profile</a>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card h-100">
                            <div class="card-body text-center">
                                <h5 class="card-title">💬 Chat with Your Data</h5>
                                <p class="card-text">Ask questions about your data in plain English and get instant answers.</p>
                                <a href="/chat/{file_id}" class="btn btn-info" target="_blank">Start Chat Session</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
    </html>
    """)

# ==============================================================================
# 4. BACKEND - Core Logic for Uploading and Report Generation
# ==============================================================================
@app.post("/upload/")
async def upload_and_process(file: UploadFile = File(...)):
    """Handles file upload, cleaning, caching, and redirects to the dashboard."""
    contents = await file.read()
    df = pd.read_csv(io.StringIO(str(contents, 'utf-8')))
    df = df.dropna().drop_duplicates()

    file_id = str(uuid.uuid4())
    df_cache[file_id] = df
    return RedirectResponse(url=f"/dashboard/{file_id}", status_code=303)

@app.get("/ai_visuals/{file_id}")
async def generate_ai_visuals(file_id: str):
    """Generates the AI-powered visualization report."""
    if file_id not in df_cache: return RedirectResponse(url="/")
    df = df_cache[file_id]
    
    # AI Logic to generate code
    data_summary = f"Dataset has {df.shape[0]} rows, Numerical columns: {df.select_dtypes(include=['number']).columns.tolist()}, Categorical columns: {df.select_dtypes(include=['object', 'category']).columns.tolist()}"
    prompt = f"You are a visualization expert using Plotly Express. Based on this summary, provide a JSON list of 3 objects. Each object must have 'title' and 'code' (a single line of Plotly Express code). The DataFrame is named 'df'. Dataset Summary: {data_summary}"
    
    try:
        ai_response = ai_model.generate_content(prompt)
        cleaned_response = ai_response.text.strip().replace("```json", "").replace("```", "")
        charts_to_generate = json.loads(cleaned_response)
    except Exception: return HTMLResponse(f"<h1>Error processing AI response.</h1><pre>{ai_response.text}</pre>")
    
    all_charts_html = ""
    for chart_info in charts_to_generate:
        try:
            fig = eval(chart_info["code"])
            all_charts_html += f'<h3>{chart_info["title"]}</h3>{fig.to_html(full_html=False, include_plotlyjs="cdn")}'
        except Exception as e: all_charts_html += f"<h3>Error generating chart: {chart_info['title']}</h3><p>{e}</p>"
    
    return HTMLResponse(content=f"<html><body><h1>AI Visualizations</h1>{all_charts_html}</body></html>")

@app.get("/statistical_report/{file_id}")
async def generate_statistical_report(file_id: str):
    """Generates the detailed ydata-profiling statistical report."""
    from ydata_profiling import ProfileReport
    if file_id not in df_cache: return RedirectResponse(url="/")
    df = df_cache[file_id]
    profile = ProfileReport(df, title="Statistical Profile")
    return HTMLResponse(content=profile.to_html())

# ==============================================================================
# 5. CHAT FUNCTIONALITY
# ==============================================================================
@app.get("/chat/{file_id}")
def chat_page(file_id: str):
    """Serves the main chat interface page."""
    if file_id not in df_cache: return RedirectResponse(url="/")
    # This HTML contains the JavaScript to handle the chat interaction
    return HTMLResponse(content="""
        <!DOCTYPE html><html><head><title>Chat</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"></head>
        <body><div class="container mt-4"><h2>Chat with your AI Data Analyst</h2>
        <div id="chat-box" class="border p-3 rounded" style="height: 400px; overflow-y: scroll;"></div>
        <form id="chat-form" class="mt-3"><div class="input-group"><input type="text" id="question" class="form-control" placeholder="e.g., How many rows are there?" required><button class="btn btn-primary" type="submit">Send</button></div></form>
        </div><script>
        const chatForm = document.getElementById('chat-form'), chatBox = document.getElementById('chat-box'), qInput = document.getElementById('question');
        chatForm.addEventListener('submit', async(e) => {
            e.preventDefault(); const q = qInput.value; if (!q) return;
            chatBox.innerHTML += `<p><b>You:</b> ${q}</p>`; qInput.value = '';
            const response = await fetch(window.location.pathname.replace('/chat/', '/ask/'), {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: q})
            });
            const data = await response.json();
            chatBox.innerHTML += `<p><b>AI:</b><br>${data.answer.replace(/\\n/g, '<br>')}</p>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        });
        </script></body></html>
    """)

@app.post("/ask/{file_id}")
async def ask_question(file_id: str, item: ChatQuestion):
    """Handles a user's question, gets code from the AI, executes it, and returns the answer."""
    if file_id not in df_cache: return {"answer": "Session not found."}
    df = df_cache[file_id]
    prompt = f"You are a Python Pandas expert. Given a DataFrame named 'df' with columns {df.columns.tolist()}, write a single line of Python code to answer: '{item.question}'. Your code must print the result. No explanation."
    try:
        ai_response = ai_model.generate_content(prompt)
        code = ai_response.text.strip().replace("```python", "").replace("```", "")
        output_stream = io.StringIO()
        exec(code, {'df': df, 'pd': pd}, {'__builtins__': {}, 'print': lambda *a, **k: print(*a, file=output_stream, **k)})
        answer = output_stream.getvalue() or "Action performed."
    except Exception as e: answer = f"Error: {e}"
    return {"answer": answer}