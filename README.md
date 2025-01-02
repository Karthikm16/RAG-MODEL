# RAG Model with LangChain and Streamlit

This is a Retrieval Augmented Generation (RAG) implementation using LangChain and Ollama's Mistral model, with a Streamlit interface for document processing and question answering.

## Features

- Upload and process multiple document types (PDF, DOCX, TXT)
- Document chunking and embedding using sentence-transformers
- Vector storage using Chroma DB
- Question answering using Mistral model through Ollama
- User-friendly Streamlit interface

## Prerequisites

1. Install [Ollama](https://ollama.ai/)
2. Pull the Mistral model:
   ```
   ollama pull mistral
   ```

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. Upload your documents using the file uploader
2. Click "Process Documents" to analyze them
3. Ask questions about your documents in the text input
4. Get AI-generated answers based on the document content

## Technical Details

- Uses HuggingFace's sentence-transformers for document embedding
- Implements RecursiveCharacterTextSplitter for optimal document chunking
- Utilizes Chroma as the vector store
- Leverages Mistral model through Ollama for generation
- Implements RAG pattern for accurate, context-aware responses
