import streamlit as st
import time
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
from typing import TypedDict, Sequence
from langgraph.graph import StateGraph, END

# --- LOAD SECRETS ---
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ContextMind AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS INJECTION ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { font-weight: 600 !important; letter-spacing: -0.02em; }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- FUNCTIONS FOR RAG ---
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_txt_text(txt_files):
    text = ""
    for txt in txt_files:
        text += txt.getvalue().decode("utf-8")
    return text

def get_youtube_text(url):
    try:
        video_id = ""
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            return ""
        
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        text = " ".join([t.text for t in transcript])
        return text + "\n\n"
    except Exception as e:
        st.sidebar.error(f"Could not extract YouTube transcript: {e}")
        return ""

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# --- INITIALIZE LLM BRAIN ---
llm_engine_sidebar = "gpt-4o-mini" 
llm = ChatOpenAI(model=llm_engine_sidebar, temperature=0.2)

# --- SIDEBAR (App Controls) ---
with st.sidebar:
    st.title("⚙️ Control Center")
    st.markdown("Configure your Agentic Second Brain.")
    st.divider()
    
    st.selectbox("LLM Engine", [llm_engine_sidebar], disabled=True)
    temperature = st.slider("Reasoning Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    llm = ChatOpenAI(model=llm_engine_sidebar, temperature=temperature)
    
    st.divider()
    st.markdown("### 📂 Knowledge Base")
    
    uploaded_files = st.file_uploader("Upload Documents (PDF, TXT)", accept_multiple_files=True, type=['pdf', 'txt'])
    youtube_url = st.text_input("Add YouTube Video URL")
    
    if st.button("Process Knowledge", use_container_width=True):
        if uploaded_files or youtube_url:
            with st.spinner("Crunching vectors..."):
                raw_text = ""
                
                if uploaded_files:
                    pdfs = [f for f in uploaded_files if f.name.endswith('.pdf')]
                    txts = [f for f in uploaded_files if f.name.endswith('.txt')]
                    raw_text += get_pdf_text(pdfs) + get_txt_text(txts)
                
                if youtube_url:
                    raw_text += get_youtube_text(youtube_url)
                
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = create_vector_store(text_chunks)
                    st.success("Knowledge Base Built & Ready for Agentic Routing!")
        else:
            st.warning("Please upload a file or paste a URL first.")
            
    st.divider()
    if st.button("Clear Active Memory", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.caption("ContextMind AI v2.0 | LangGraph Agent Mode")

# ==========================================
# --- LANGGRAPH AGENT ARCHITECTURE ---
# ==========================================

# 1. Define the State (The memory the agent passes around)
class AgentState(TypedDict):
    messages: list
    route: str
    context: str

# 2. Node: The Router (Decides what to do)
def router_node(state: AgentState):
    user_query = state["messages"][-1]["content"]
    
    # If no data is uploaded yet, just chat normally
    if "vector_store" not in st.session_state:
        return {"route": "chat_node"}
    
    # Ask the LLM to classify the intent
    prompt = f"You are a routing assistant. Does this user query require searching through specific uploaded documents or videos? Query: '{user_query}'. Reply strictly with 'SEARCH' or 'CHAT'."
    decision = llm.invoke(prompt).content.strip().upper()
    
    if "SEARCH" in decision:
        return {"route": "retriever_node"}
    return {"route": "chat_node"}

# 3. Node: The Retriever (Pulls vectors)
def retriever_node(state: AgentState):
    user_query = state["messages"][-1]["content"]
    docs = st.session_state.vector_store.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

# 4. Node: The Synthesizer (Answers using documents)
def synthesizer_node(state: AgentState):
    context = state.get("context", "")
    sys_msg = SystemMessage(content=f"You are ContextMind AI. Use this retrieved context to answer the user accurately:\n{context}")
    
    langchain_msgs = [sys_msg]
    for msg in state["messages"]:
        if msg["role"] == "user": langchain_msgs.append(HumanMessage(content=msg["content"]))
        else: langchain_msgs.append(AIMessage(content=msg["content"]))
        
    response = llm.invoke(langchain_msgs).content
    return {"messages": state["messages"] + [{"role": "assistant", "content": response}]}

# 5. Node: Normal Chat (Answers without documents)
def chat_node(state: AgentState):
    sys_msg = SystemMessage(content="You are ContextMind AI, a highly advanced agentic second brain built by a 180 IQ engineer.")
    
    langchain_msgs = [sys_msg]
    for msg in state["messages"]:
        if msg["role"] == "user": langchain_msgs.append(HumanMessage(content=msg["content"]))
        else: langchain_msgs.append(AIMessage(content=msg["content"]))
        
    response = llm.invoke(langchain_msgs).content
    return {"messages": state["messages"] + [{"role": "assistant", "content": response}]}

# 6. Edge Logic
def route_edge(state: AgentState):
    return state["route"]

# 7. Compile the Graph
workflow = StateGraph(AgentState)
workflow.add_node("router_node", router_node)
workflow.add_node("retriever_node", retriever_node)
workflow.add_node("synthesizer_node", synthesizer_node)
workflow.add_node("chat_node", chat_node)

workflow.set_entry_point("router_node")
workflow.add_conditional_edges("router_node", route_edge, {"retriever_node": "retriever_node", "chat_node": "chat_node"})
workflow.add_edge("retriever_node", "synthesizer_node")
workflow.add_edge("synthesizer_node", END)
workflow.add_edge("chat_node", END)

agent_app = workflow.compile()

# ==========================================
# --- STREAMLIT UI ---
# ==========================================

st.title("🧠 ContextMind AI")
st.markdown("#### The Agentic Second Brain")
st.caption("Powered by LangGraph Autonomous Routing.")
st.divider()

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Welcome to ContextMind AI. My Neo4j constraints are removed, and I am now running pure LangGraph Agentic Routing. Try asking me a general question, then upload a document and ask a specific one to watch my router adapt!"
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query your knowledge base..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # We use st.status to visually show the user the Agent's thought process!
        with st.status("🧠 Agent thinking...", expanded=True) as status:
            st.write("Analyzing intent...")
            
            # Run the LangGraph Agent
            final_state = agent_app.invoke({"messages": st.session_state.messages, "route": "", "context": ""})
            
            # Print the steps it took
            if final_state["route"] == "retriever_node":
                st.write("📂 Intent classified: Document Search Required.")
                st.write("🔍 Searching Vector Database...")
                st.write("✍️ Synthesizing Answer...")
            else:
                st.write("💬 Intent classified: General Chat.")
                st.write("⚡ Accessing Base LLM Memory...")
                
            status.update(label="Response generated!", state="complete", expanded=False)
            
        ai_answer = final_state["messages"][-1]["content"]
        st.markdown(ai_answer)
        
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})