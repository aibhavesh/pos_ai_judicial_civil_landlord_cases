# app.py

import streamlit as st
import os
import nest_asyncio

# This is needed for Streamlit to run LangChain's async functions
nest_asyncio.apply()

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Judicial Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SECURE API KEY SETUP ---
st.sidebar.title("Configuration")
api_key = None
try:
    # This works when deployed on Streamlit Community Cloud
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("API Key loaded from secrets!", icon="✅")
except:
    # A fallback for local development if secrets.toml is not found
    st.sidebar.warning("Could not find Streamlit secret. Please provide your key.")
    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")

if not api_key:
    st.info("Please provide your Google API Key to proceed.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# --- 4. CACHED FUNCTION TO LOAD THE RAG ENGINE ---
# @st.cache_resource ensures this complex function runs only once.
@st.cache_resource
def load_resources():
    try:
        folder_path = 'poc_civil_cases'
        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            return "Error: The 'poc_civil_cases' directory is empty or not found. Please add your .txt case files."

        # Read all text files
        case_documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    case_documents.append(file.read())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.create_documents(case_documents)

        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)

        # Initialize the LLM
        llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
        
        return {"llm": llm, "vector_store": vector_store}

    except Exception as e:
        return f"An error occurred while loading resources: {e}"

# Load resources and handle potential errors
resources = load_resources()
if isinstance(resources, str):
    st.error(resources)
    st.stop()
llm = resources["llm"]
vector_store = resources["vector_store"]


# --- 5. USER INTERFACE ---
st.title("⚖️ AI Judicial Assistant POC")
st.markdown("This tool analyzes your civil case document against a knowledge base of past judgments to provide a strategic report.")

with st.container(border=True):
    st.subheader("1. Enter Case Details")
    user_goal = st.text_input("Your Goal in this Case:", placeholder="e.g., 'To defend against an eviction notice'")
    user_role = st.selectbox("Your Role:", ["Advocate for Plaintiff", "Advocate for Defendant", "Judge"])
    case_document_text = st.text_area("Paste the Case Document Text Here:", height=300, placeholder="Paste the full plaint, notice, or written statement...")

# --- 6. MAIN LOGIC AND OUTPUT (Corrected Version) ---
if st.button("Generate Legal Analysis Report", type="primary"):
    if not all([user_goal, user_role, case_document_text]):
        st.warning("Please fill in all the fields.")
    else:
        with st.spinner("Analyzing... This may take a moment."):
            try:
                # This is the corrected master prompt template.
                # It expects a single 'question' variable (which we will format) and a 'context' variable.
                MODIFIED_LEGAL_PROMPT = """
                You are an Indian Legal AI Assistant. Your task is to analyze the following case information, match it with relevant legal precedents from the provided context, and prepare a strategic report.

                **CASE INFORMATION:**
                {question}

                **CONTEXT FROM SIMILAR JUDGMENTS:**
                {context}

                **INSTRUCTIONS:**
                Based on all the information above, generate a comprehensive legal analysis report by strictly following these steps:

                **Step 1: CASE EXTRACTION**
                - Summarise the key facts, timeline, and parties involved from the "CASE INFORMATION".
                - Identify the type of case.

                **Step 2: LEGAL RELEVANCE**
                - Suggest relevant sections from Indian laws.
                - Briefly explain why each section is relevant.

                **Step 3: CASE MATCHING (Analysis of Context)**
                - Analyze the "CONTEXT FROM SIMILAR JUDGMENTS" provided.
                - Explain how points from the context support or weaken the current case.

                **Step 4: LEGAL STRATEGY & ARGUMENT SIMULATION (Based on the User's Role)**
                - **Likely Questions from the Judge:** List 3-4 pointed questions.
                - **Possible Arguments from Opposing Counsel:** List 2-3 strong arguments.
                - **Suggested Counter-Arguments:** Provide robust counters.
                - **Supportive Evidence:** Suggest key documents or witness testimonies needed.

                **Step 5: PERMUTATION OF ARGUMENTS & RISK ANALYSIS**
                - Outline 2-3 possible legal strategies.
                - For each strategy, mention potential outcomes and risks.

                **Step 6: FINAL OUTPUT**
                - Structure the entire analysis into a professional report with these headings:
                  A. Case Summary
                  B. Relevant Laws & Sections
                  C. Analysis of Similar Judgments
                  D. Strategic Q&A and Evidence
                  E. Final Recommendations and Risk Analysis
                """
                
                prompt = PromptTemplate(
                    template=MODIFIED_LEGAL_PROMPT,
                    input_variables=["context", "question"]
                )
                
                # Combine all user inputs into a single variable for the prompt
                combined_input = f"""
                **User’s Goal:** {user_goal}
                **User's Role:** {user_role}
                **Case Document Text:** {case_document_text}
                """

                # Set up the RetrievalQA chain
                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(),
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
                
                # Invoke the chain with the single combined input string
                report_result = rag_chain.invoke(combined_input)

                st.success("Analysis Complete!")
                st.subheader("Generated Legal Strategy Report")
                st.markdown(report_result['result'])

                with st.expander("View Retrieved Source Documents Used for Analysis"):
                    for doc in report_result['source_documents']:
                        st.markdown(f"---")
                        st.write(doc.page_content[:500] + "...")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")