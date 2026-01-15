import streamlit as st
import os
from logic import extract_pdf_data, analyze_report_with_llm, setup_vectorstore, generate_medical_analysis

st.set_page_config(page_title="Report Analyzer", layout="wide")

st.title("🩸 Blood Report AI Assistant")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    # THE CHOICE BUTTON
    analysis_mode = st.radio(
        "Choose Analysis Mode:",
        ("Strict (RAG Only)", "Lenient (AI Assisted)"),
        help="Strict mode will not answer if info is missing from your files. Lenient mode allows the LLM to use its own training data for context."
    )

gemini_key = os.getenv('YOUR_GEMINI_API_KEY')
hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if not gemini_key or not hf_token:
    st.error("API keys missing in .env")
    st.stop()

uploaded_file = st.file_uploader("Upload Blood Report (PDF)", type="pdf")

if uploaded_file:
    # Convert display name to function argument
    mode_arg = "Strict" if "Strict" in analysis_mode else "Lenient"
    
    if st.button(f"Generate {mode_arg} Analysis"):
        with st.spinner(f"Running {mode_arg} Analysis..."):
            _, text = extract_pdf_data(uploaded_file)
            analysis_output = analyze_report_with_llm(text, gemini_key)
            v_store = setup_vectorstore()
            
            # Pass the chosen mode to the logic function
            final_report = generate_medical_analysis(analysis_output, v_store, gemini_key, mode=mode_arg)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Patient Details")
                st.write(analysis_output['Patient_info'])
                st.subheader("Detected Issues")
                for issue in analysis_output['Issues']:
                    st.warning(issue)
            
            with col2:
                st.subheader(f"Analysis Report ({mode_arg} Mode)")
                st.markdown(final_report)