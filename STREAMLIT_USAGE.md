# Streamlit Usage Guide

This project uses **Streamlit** to provide a simple web UI for uploading a PDF and running the analysis pipeline.

## How to Run the Streamlit App

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Set environment variables**
Create a `.env` file in the project root:
```env
YOUR_GEMINI_API_KEY=your_gemini_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

3. **Start the Streamlit server**
```bash
streamlit run app.py
```

4. **Use the UI**
- Open the local URL shown in the terminal (usually `http://localhost:8501`)
- Upload a blood report PDF
- Select **Strict (RAG Only)** or **Lenient (AI Assisted)** mode
- Click **Generate Analysis** to view results

## What Streamlit Handles Here
- File upload (PDF)
- Mode selection (Strict vs Lenient)
- Triggering backend logic
- Displaying patient details, detected issues, and final analysis
