# 🧠 ContextMind AI

**An Autonomous, Multi-Modal Second Brain powered by LangGraph and Agentic RAG.**

ContextMind AI is not a standard chatbot. It is a state-of-the-art **Agentic Retrieval-Augmented Generation (RAG)** system designed to ingest complex multi-modal data (PDFs and YouTube Transcripts) and autonomously route reasoning tasks using a localized graph workflow.

Built to showcase end-to-end AI engineering, this project demonstrates a shift from linear RAG pipelines to autonomous, state-driven agent architectures.

---

### 🚀 Technical Architecture

At the core of ContextMind AI is **LangGraph**, which manages the system's cognitive state and decision-making framework. Instead of blindly querying a database, the application utilizes an LLM-powered **Router Node** to classify user intent in real-time.

* **The Router:** Evaluates if a query requires deep document retrieval or general contextual conversation.
* **The Retriever:** Interfaces mathematically with a local **FAISS Vector Database** using OpenAI Embeddings to extract precise contextual chunks.
* **The Synthesizer:** Compiles the extracted vectors and chat history, feeding them through GPT-4o-mini to generate highly accurate, hallucination-free responses.
* **The Memory Layer:** Maintains persistent session states, allowing for multi-turn conversations and contextual follow-ups.

---

### ⚙️ Core Features

* **Multi-Modal Ingestion:** Dynamically extracts and parses text from uploaded PDFs and live YouTube video URLs (via transcript APIs).
* **Agentic Routing:** Visually demonstrates autonomous decision-making, showing the user exactly how the agent is classifying and routing their query.
* **Mathematical Context Extraction:** Utilizes `RecursiveCharacterTextSplitter` and high-dimensional embeddings to find meaning, not just keywords.
* **Professional UI:** Built with a custom-styled, minimalist Streamlit interface designed for a premium user experience.

---

### 🛠️ Tech Stack

* **Orchestration & Agents:** LangChain, LangGraph
* **Language Models & Embeddings:** OpenAI (GPT-4o-mini, text-embedding-3)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Data Ingestion:** PyPDF, YouTube Transcript API
* **Frontend:** Streamlit (with custom CSS injection)
* **Environment Management:** Python `venv`, python-dotenv

---

### 💻 Run Locally

To spin up the agentic workspace on your local machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/Vinesh96000/Contextmind-AI.git](https://github.com/Vinesh96000/Contextmind-AI.git)
   cd Contextmind-AI

2. Activate a Python virtual environment and install the arsenal:

   pip install -r requirements.txt
   Configure your environment variables by creating a .env file:

3. OPENAI_API_KEY=sk-your-api-key-here
   Launch the application:

4. streamlit run app.py   




👨‍💻 About the Developer
Built by Vinesh J. Aspiring AI & Machine Learning Engineer
Passionate about end-to-end AI development, model architecture, and building systems that bridge the gap between complex machine learning concepts and highly functional, real-world applications.
