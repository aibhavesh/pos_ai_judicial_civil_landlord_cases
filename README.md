# ⚖️ AI Judicial Assistant — POC for Civil & Landlord Cases

> A **Proof of Concept** AI-powered legal analysis tool that uses Retrieval-Augmented Generation (RAG) to help advocates and judges analyse civil & landlord cases against a knowledge base of past Indian judgments.

---

## 🚀 Live Demo

Deploy instantly on **Streamlit Community Cloud** — just add your Google API key and go.

---

## 📌 What It Does

Paste any civil case document (plaint, notice, written statement, etc.) into the app and receive a **structured, AI-generated legal strategy report** in seconds.

The report covers:

| Section | Description |
|---|---|
| **Case Summary** | Key facts, timeline, and parties extracted from your document |
| **Relevant Laws & Sections** | Applicable Indian statutes with explanations |
| **Analysis of Similar Judgments** | How past rulings support or weaken your case |
| **Strategic Q&A and Evidence** | Likely judge questions, opposing arguments, counter-arguments & key evidence |
| **Final Recommendations & Risk Analysis** | 2–3 legal strategies with projected outcomes and risks |

---

## 🧠 How It Works — RAG Architecture

```
User Input (Case Document)
        │
        ▼
  Google Gemini 1.5 Flash  ◄──────────────────────────────────┐
        │                                                       │
        ▼                                                       │
  Combined Prompt                                              │
  (Goal + Role + Document)                                     │
        │                                                       │
        ▼                                                   FAISS
  RetrievalQA Chain ──── Vector Similarity Search ──► Past Case Chunks
        │                  (Google Embeddings)            (15 POC cases)
        ▼
  Structured Legal Strategy Report
```

1. **Indexing** — On startup, all `.txt` files in `poc_civil_cases/` are loaded, split into 1,500-token chunks (200-token overlap), and embedded with `models/embedding-001` into a FAISS in-memory vector store.
2. **Retrieval** — When the user submits a case, the query is embedded and the most semantically similar past-case chunks are retrieved.
3. **Generation** — The retrieved chunks (context) and the user's case details are fed into a carefully engineered prompt template, and **Gemini 1.5 Flash** generates the full report.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | [Streamlit](https://streamlit.io) |
| **LLM** | Google Gemini 1.5 Flash (`langchain-google-genai`) |
| **Embeddings** | Google `embedding-001` model |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (in-memory) |
| **Orchestration** | [LangChain](https://www.langchain.com/) (`RetrievalQA`) |
| **Async Support** | `nest_asyncio` (required for Streamlit ↔ LangChain) |
| **Knowledge Base** | 15 real Indian civil & landlord case documents (`.txt`) |

---

## 📁 Project Structure

```
pos_ai_judicial_civil_landlord_cases/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level dependencies (ffmpeg, libsm6, libxext6)
│
├── poc_civil_cases/        # Knowledge base — 15 past Indian case judgments
│   ├── case1.txt
│   ├── case2.txt
│   └── ... (case3 – case15)
│
└── .streamlit/
    └── secrets.toml        # API key storage (not committed to git)
```

---

## ⚡ Getting Started

### Prerequisites

- Python 3.9+
- A **Google AI Studio API Key** (free tier available at [aistudio.google.com](https://aistudio.google.com))

### 1. Clone the Repository

```bash
git clone https://github.com/aibhavesh/pos_ai_judicial_civil_landlord_cases.git
cd pos_ai_judicial_civil_landlord_cases
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your API Key

**Option A — Streamlit secrets (recommended):**

Create the file `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-google-api-key-here"
```

**Option B — Enter it in the sidebar at runtime** (no file needed).

### 4. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Using the App

1. **Enter your goal** — e.g., *"To defend against an eviction notice"*
2. **Select your role** — Advocate for Plaintiff / Advocate for Defendant / Judge
3. **Paste the case document** — full plaint, notice, written statement, etc.
4. Click **"Generate Legal Analysis Report"**
5. Review the structured report and expand the **"View Retrieved Source Documents"** section to see which past cases influenced the analysis.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Fork this repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your fork.
3. In the app settings → **Secrets**, add:
   ```
   GOOGLE_API_KEY = "your-google-api-key-here"
   ```
4. Click **Deploy** — done!

---

## 🔮 Roadmap / Future Improvements

- [ ] Add PDF upload support for case documents
- [ ] Persist the FAISS index to disk to avoid re-indexing on every restart
- [ ] Expand the knowledge base beyond the 15 POC cases
- [ ] Add multi-language support (Hindi, Marathi, etc.)
- [ ] Integrate a citation tracker to reference exact judgment sections
- [ ] Fine-tune an LLM specifically on Indian legal corpus

---

## ⚠️ Disclaimer

This tool is a **Proof of Concept** intended for research and educational purposes only. It does **not** constitute legal advice. Always consult a qualified legal professional for actual legal matters.

---

## 📄 License

This project is open-source. Feel free to fork, star ⭐, and contribute!

---

*Built with ❤️ using Streamlit, LangChain, and Google Gemini.*
