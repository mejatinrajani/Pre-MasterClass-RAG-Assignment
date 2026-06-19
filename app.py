import streamlit as st
import logging
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.synthesizer import RAGSynthesizer
from src.config import Config

# Set up basic logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Bastian Beach Club RAG", layout="wide")

@st.cache_resource
def initialize_pipeline():
    """Initialize backend components once."""
    with st.spinner("Initializing Knowledge Base and Databases..."):
        # The HybridRetriever automatically invokes the MockDatabase singleton now!
        retriever = HybridRetriever()
        synthesizer = RAGSynthesizer()
        return retriever, synthesizer

# Initialize backend
if "retriever" not in st.session_state or "synthesizer" not in st.session_state:
    retriever, synthesizer = initialize_pipeline()
    st.session_state.retriever = retriever
    st.session_state.synthesizer = synthesizer

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Bastian Beach Club - AI Assistant")

# Sidebar configuration and debug info
with st.sidebar:
    st.header("Pipeline Diagnostics")
    st.write(f"**Fast LLM:** {Config.FAST_LLM}")
    st.write(f"**Reasoning LLM:** {Config.REASONING_LLM}")
    st.write(f"**Top-K Retrieval:** {Config.TOP_K_RETRIEVAL}")
    
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "diagnostics" in msg:
            with st.expander("View Retrieval Diagnostics"):
                st.json(msg["diagnostics"])

# Chat input
if prompt := st.chat_input("Ask about Bastian Beach Club policies, menu, or events..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Querying Hybrid Knowledge Base..."):
            # 1. Retrieve Context
            retrieval_result = st.session_state.retriever.retrieve(prompt)
            
            # 2. Synthesize Answer
            generation_result = st.session_state.synthesizer.generate_answer(prompt, retrieval_result,st.session_state.messages)
            
            answer = generation_result["answer"]
            engine_used = generation_result["engine_used"]
            
            st.markdown(answer)
            
            # Diagnostics to show the graph fallback
            diagnostics = {
                "Engine Triggered": engine_used,
                "Entities Extracted": retrieval_result.get("entities_extracted", []),
                "Chunks Retrieved": len(retrieval_result.get("context_chunks", [])),
                "Context Preview": [c["text"] for c in retrieval_result.get("context_chunks", [])][:2]
            }
            
            with st.expander("View Retrieval Diagnostics"):
                st.json(diagnostics)
                
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "diagnostics": diagnostics
    })