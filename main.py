import pdfplumber
import google.generativeai as genai
import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_pdf_data(file_path='report.pdf'):
    all_tables = []
    all_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            all_tables.extend(tables)
            all_text += page.extract_text() + "\n"
    return all_tables, all_text

def get_api_key():
    return os.getenv('YOUR_GEMINI_API_KEY')

def analyze_report_with_llm(all_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    chunk_size = 10000 
    chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]

    metric_text = ""
    for chunk in chunks:
        prompt = f"""Analyze the following part of a blood test report. \              
        1. Extract patient information: Name, Age, Sex, Printed On date.\               
        2. Identify all important metrics and list them as 'Low [test]' or 'High [test]'\                   
        or 'Good' for all the value ranges present in the blood report. \               
        Report text: {chunk}"""
        
        response = model.generate_content(prompt)
        metric_text += response.text.strip() + "\n"

    metrices = []
    issues = []
    patient_info = {}

    lines = metric_text.split('\n')
    for line in lines:
        if 'Low' in line or 'High' in line or 'Good' in line:
            metrices.append(line.strip())
        if 'Low' in line or 'High' in line:
            issues.append(line.strip())
        elif 'Name:' in line:
            patient_info['name'] = line.split(':', 1)[1].strip()
        elif 'Age:' in line:
            patient_info['age'] = line.split(':', 1)[1].strip()      
        elif 'Sex:' in line:
            patient_info['sex'] = line.split(':', 1)[1].strip()
        elif 'Printed On:' in line or 'Date:' in line:
            patient_info['Date'] = line.split(':', 1)[1].strip()

    output = {'Patient_info': patient_info, 'Metrices': metrices, 'Issues': issues}
    return output

def setup_vectorstore():
    token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=token,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    query = "what happens if Glucose count is low in blood test"
    vectorstore.similarity_search(query, k=5)
    return vectorstore

def generate_medical_analysis(output, vectorstore, api_key):
    issues_text = "\n".join(output["Issues"])
    
    prompt = f"""
    You are a medical report analysis assistant.

    ### Context
    You are given extracted blood test abnormalities from a patient report.
    Use ONLY the information available in the Vector Store to answer.
    Do NOT use prior knowledge or assumptions.
    If relevant information is missing, explicitly say:
    "I am sorry, I do not have enough information to provide an answer."

    ### Identified Blood Test Issues
    {issues_text}

    ### Instructions
    For EACH clearly identified abnormal parameter:
    1. Explain what the parameter indicates (in simple medical terms)
    2. List possible causes ONLY if present in the Vector Store
    3. Suggest general remedies or lifestyle changes ONLY if present in the Vector Store
    4. Avoid giving diagnoses, medications, or dosages
    5. If reference ranges or patient context are missing, explicitly mention the uncertainty

    ### Output Format
    Return the answer in the following structure:

    #### [Parameter Name]
    - **What it means:**
    - **Possible causes (from Vector Store):**
    - **General remedies / precautions (from Vector Store):**
    - **Confidence level:** High / Medium / Low (based on available context)

    ### Safety Rules
    - Do NOT hallucinate medical facts
    - Do NOT combine multiple issues unless the Vector Store explicitly links them
    - If an issue appears to be metadata, duplicated, or inconclusive, clearly state that

    ### Closing
    End with a brief note stating that this is informational and not a medical diagnosis.
    """

    docs = vectorstore.similarity_search(prompt, k=5)
    
    context_from_docs = "".join([doc.page_content for doc in docs])
    final_prompt = prompt + context_from_docs
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    response = model.generate_content(final_prompt)
    return response.text.strip()

def save_to_markdown(content):
    
    md_content = f"# Blood Report Analysis\n\n{content}\n"
    with open("blood_report_analysis.md", "w", encoding="utf-8") as f:
        f.write(md_content)

if __name__ == "__main__":
    # Execute workflow
    tables, text = extract_pdf_data('report.pdf')
    api_key = get_api_key()
    analysis_output = analyze_report_with_llm(text, api_key)
    v_store = setup_vectorstore()
    final_analysis = generate_medical_analysis(analysis_output, v_store, api_key)
    save_to_markdown(final_analysis)
    print("Markdown file saved as blood_report_analysis.md")