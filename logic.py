import pdfplumber
import google.generativeai as genai
import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

def extract_pdf_data(uploaded_file):
    all_tables = []
    all_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            all_tables.extend(tables)
            all_text += page.extract_text() + "\n"
    return all_tables, all_text

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

    metrices, issues, patient_info = [], [], {}
    lines = metric_text.split('\n')
    for line in lines:
        if any(word in line for word in ['Low', 'High', 'Good']):
            metrices.append(line.strip())
        if 'Low' in line or 'High' in line:
            issues.append(line.strip())
        elif 'Name:' in line: patient_info['name'] = line.split(':', 1)[1].strip()
        elif 'Age:' in line: patient_info['age'] = line.split(':', 1)[1].strip()
        elif 'Sex:' in line: patient_info['sex'] = line.split(':', 1)[1].strip()
        elif 'Printed On:' in line or 'Date:' in line: patient_info['Date'] = line.split(':', 1)[1].strip()

    return {'Patient_info': patient_info, 'Metrices': metrices, 'Issues': issues}

def setup_vectorstore():
    token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=token,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def generate_medical_analysis(output, vectorstore, api_key, mode="Strict"):
    issues_text = "\n".join(output["Issues"])
    
    if mode == "Strict":
        instructions = """
        STRICT MODE: Use ONLY the information available in the Vector Store. 
        Do NOT use prior knowledge. If info is missing, say 'Information not found in database'.
        """
    else:
        instructions = """
        LENIENT MODE: Use the Vector Store as your primary source, but you may provide 
        general medical context or common knowledge for better clarity if the database is silent.
        """

    prompt = f"""
    {instructions}
    ### Identified Blood Test Issues:
    {issues_text}
    
    For EACH issue:
    1. Explain the parameter.
    2. List causes (Strictly from Store if in Strict mode).
    3. Suggest remedies.
    """
    
    docs = vectorstore.similarity_search(prompt, k=5)
    context = "".join([doc.page_content for doc in docs])
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    response = model.generate_content(prompt + "\n\nCONTEXT FROM DATABASE:\n" + context)
    return response.text.strip()