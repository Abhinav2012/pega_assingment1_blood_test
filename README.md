# Blood Report Analysis using RAG + LLM

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to analyze blood test reports in PDF format using **Large Language Models (LLMs)** and a **vector database**. The system combines automated **PDF extraction**, **LLM-based medical signal detection**, and a **document ingestion pipeline** to generate safe, grounded medical explanations.

---

## System Architecture (High-Level)

1. PDF data extraction  
2. LLM-based metric & patient info extraction  
3. Knowledge ingestion & vectorization  
4. Vector store persistence (ChromaDB)  
5. Retrieval-Augmented Generation (RAG)  
6. Markdown report generation  

---

## PDF Extraction Layer

- Implemented using `pdfplumber`
- Extracts:
  - Free-form text from each page
  - Tables (if present)
- All extracted text is merged into a single corpus

This extracted text is passed directly to the LLM for medical metric identification.

---

## LLM: Medical Information Extraction

- Powered by **Google Gemini (`gemini-2.5-flash-lite`)**
- Handles large reports by chunking input text
- Extracts:
  - Patient metadata (Name, Age, Sex, Date)
  - Blood parameters classified as **High**, **Low**, or **Good**

**Output:**
- Structured patient information
- Full list of detected metrics
- Filtered list of abnormal parameters (Issues)

This converts **unstructured medical PDFs into structured signals**.

---

## Knowledge Ingestion Pipeline (Vector Store Creation)

The ingestion pipeline (`ingestion_pipeline.py`) is responsible for building the **knowledge base** used by the RAG system.

### Key Responsibilities:
- Loads domain documents (`.txt`) from the `docs/` directory
- Splits documents into overlapping chunks
- Generates embeddings using HuggingFace models
- Persists embeddings into **ChromaDB**

### Ingestion Flow:
1. Load `.txt` files via `DirectoryLoader`
2. Split content using `CharacterTextSplitter`
3. Generate embeddings with:
   - `sentence-transformers/all-MiniLM-L6-v2`
4. Store vectors persistently in `./chroma_db`

This step ensures **all medical explanations are grounded in explicitly provided documents**.

---

## Vector Store (Retrieval Layer)

- Uses **ChromaDB** for semantic search
- Persisted locally for reuse
- Serves as the **single source of truth**

The LLM is strictly constrained to retrieve and reason **only** from this vector store.

---

## RAG: Retrieval-Augmented Generation

This is the core intelligence layer.

### How It Works:
1. Abnormal blood parameters are extracted
2. Relevant context is retrieved from the vector store
3. The LLM generates explanations using retrieved content only

### Safety Constraints:
- No hallucinated medical facts
- No external medical knowledge
- No diagnoses or medications
- Explicit uncertainty when context is missing

This ensures **safe, explainable, and non-speculative outputs**.

---

## Output

- Final analysis is saved as:

```
blood_report_analysis.md
```

Each abnormal parameter includes:
- What it means
- Possible causes (from vector store)
- General remedies / precautions (from vector store)
- Confidence level

---

## Environment Variables

Create a `.env` file:

```
YOUR_GEMINI_API_KEY=your_google_gemini_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

---

## Running the Project

### Step 1: Ingest Knowledge Documents
Add medical reference `.txt` files to the `docs/` folder.

```bash
python ingestion_pipeline.py
```

### Step 2: Analyze Blood Report
Ensure `report.pdf` is present.

```bash
python main.py
```
Note: Create python env using .venv
---

## Disclaimer

This project is **for informational purposes only**.  
It does **not** provide medical diagnoses or treatment advice.
