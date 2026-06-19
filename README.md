# Bastian at the Bottom - RAG AI Assistant

A Retrieval-Augmented Generation (RAG) system built with Streamlit that uses advanced hybrid retrieval to answer questions about Bastian at the Bottom policies, menu, and events.

## Features

- **Hybrid Retrieval**: Combines vector-based semantic search with knowledge graphs
- **Advanced LLM Pipeline**: Uses Groq's fast and reasoning models for intelligent responses
- **Knowledge Base**: Leverages document chunking, embeddings, and graph structures
- **Interactive UI**: Built with Streamlit for easy interaction
- **Diagnostic Tools**: Real-time pipeline diagnostics and retrieval transparency


## Getting Started

See [SETUP.md](SETUP.md) for quick setup instructions.

## Assignment Answers

See [Answers-for-the-Part-A.md](Answers-for-the-Part-A.md) for the answers of the questions asked in the provided pdf.

---
## Project Structure

```
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── src/
│   ├── config.py                   # Configuration and settings
│   ├── ingestion/
│   │   └── document_processor.py   # Document processing pipeline
│   ├── retrieval/
│   │   └── hybrid_retriever.py     # Hybrid retrieval logic
│   └── generation/
│       └── synthesizer.py          # Response generation
└── data/
    ├── raw/                        # Raw knowledge data
    ├── vector_db/                  # Vector database (Chroma)
    └── graph_db/                   # Graph database
```

## Technologies

- **Framework**: Streamlit
- **LLM Provider**: Groq (llama-3.1-8b-instant, llama-3.3-70b-versatile)
- **Vector DB**: Chroma (for embeddings)
- **Embeddings**: all-MiniLM-L6-v2
- **Language**: Python 3.8+


## API Keys

This project requires a **Groq API key**. Get one for free at [console.groq.com](https://console.groq.com).

## License

See LICENSE file for details.

---

## Author

**Name**: Jatin Rajani  
**Email**: jatin.rajani_cs23@gla.ac.in