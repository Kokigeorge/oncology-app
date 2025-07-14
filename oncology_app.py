import streamlit as st
import openai
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = "sk-abc123DEF456ghi789JKL012mno345PQR678"
  
custom_gpt_id = "g-680721e645bc8191b14df9322ac66445"

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def search_documents(query, documents):
    if not documents:
        return "No uploaded documents available."
    texts = [doc["text"] for doc in documents]
    vectorizer = TfidfVectorizer().fit_transform([query] + texts)
    cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top_match_index = cosine_similarities.argmax()
    return documents[top_match_index]["text"][:2000]

st.set_page_config(page_title="Oncology GPT App", layout="wide")
st.title("🧠 Oncology Assistant | Tumor Registry App")

site_tabs = ["Breast", "Lung", "Colon", "Prostate", "Brain", "General", "Upload New Manual"]
selected_tab = st.sidebar.radio("📂 Select Cancer Site", site_tabs)

if "docs" not in st.session_state:
    st.session_state.docs = {site: [] for site in site_tabs}

if selected_tab == "Upload New Manual":
    site_to_upload = st.selectbox("Choose Site to Upload Manual For", site_tabs[:-1])
    uploaded_file = st.file_uploader("Upload a PDF manual", type=["pdf"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.docs[site_to_upload].append({"filename": uploaded_file.name, "text": text})
        st.success(f"✅ Uploaded and indexed: {uploaded_file.name} for {site_to_upload}")
else:
    st.subheader(f"🔎 Ask a Question About: {selected_tab} Cancer")
    user_input = st.text_area("Enter your question for GPT:")
    if st.button("Ask GPT") and user_input:
        context = search_documents(user_input, st.session_state.docs[selected_tab])
        full_prompt = f"""Refer to the following medical context:\n\n{context}\n\nThen answer this question accurately and concisely:\n\n{user_input}"""
        try:
            response = openai.ChatCompletion.create(
                model=custom_gpt_id,
                messages=[
                    {"role": "system", "content": "You are a tumor registry oncology assistant."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            st.success(response["choices"][0]["message"]["content"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
