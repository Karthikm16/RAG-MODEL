import streamlit as st
import os
import tempfile
import shutil
import time
import psutil
import GPUtil
import torch
import ollama
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Any, Optional, Mapping
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
import json

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page configuration
st.set_page_config(page_title="Advanced RAG Model", layout="wide")

# Custom Ollama LLM class
class CustomOllamaLLM(LLM):
    model: str = "llama2"
    temperature: float = 0.1
    num_ctx: int = 4096
    num_gpu: int = 1
    num_thread: int = 8
    top_k: int = 30
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    gpu_layers: int = 35
    stop: List[str] = ["</s>", "Human:", "Assistant:"]

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def _llm_type(self) -> str:
        return "custom_ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_ctx': self.num_ctx,
                'num_gpu': self.num_gpu,
                'num_thread': self.num_thread,
                'top_k': self.top_k,
                'top_p': self.top_p,
                'repeat_penalty': self.repeat_penalty,
                'gpu_layers': self.gpu_layers,
                'stop': stop or self.stop
            }
        )
        return response['response']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "gpu_layers": self.gpu_layers
        }

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embeddings' not in st.session_state:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'batch_size': 32}
    )
if 'llm' not in st.session_state:
    st.session_state.llm = CustomOllamaLLM(
        model="llama2",
        temperature=0.1,
        num_ctx=4096,
        num_gpu=1,
        num_thread=8,
        top_k=30,
        top_p=0.9,
        repeat_penalty=1.1,
        gpu_layers=35
    )
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_stats' not in st.session_state:
    st.session_state.document_stats = {}
if 'key_insights' not in st.session_state:
    st.session_state.key_insights = []

def analyze_document_content(documents: List) -> Dict[str, Any]:
    """Analyze document content for insights."""
    total_text = ""
    stats = {
        'total_chars': 0,
        'total_sentences': 0,
        'sentence_lengths': [],
        'key_phrases': []
    }
    
    for doc in documents:
        text = doc.page_content
        total_text += text
        stats['total_chars'] += len(text)
        sentences = sent_tokenize(text)
        stats['total_sentences'] += len(sentences)
        stats['sentence_lengths'].extend([len(s) for s in sentences])
    
    # Extract potential key phrases (simple implementation)
    words = total_text.lower().split()
    word_freq = Counter(words)
    stats['key_phrases'] = [word for word, count in word_freq.most_common(10) if len(word) > 3]
    
    return stats

def generate_document_insights(stats: Dict[str, Any]) -> List[str]:
    """Generate insights from document statistics."""
    insights = []
    
    # Average sentence length insight
    avg_sentence_length = sum(stats['sentence_lengths']) / len(stats['sentence_lengths']) if stats['sentence_lengths'] else 0
    insights.append(f"Average sentence length: {avg_sentence_length:.1f} characters")
    
    # Document complexity insight
    if avg_sentence_length > 150:
        insights.append("Document contains complex, detailed sentences")
    elif avg_sentence_length < 50:
        insights.append("Document contains concise, brief sentences")
    
    # Key phrases insight
    insights.append(f"Top key phrases: {', '.join(stats['key_phrases'][:5])}")
    
    return insights

def log_timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        st.session_state.processing_times.append({
            'function': func.__name__,
            'duration': f"{duration:.2f}s",
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        return result
    return wrapper

@log_timing
def load_documents(uploaded_files) -> List:
    documents = []
    
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getbuffer())
            temp_path = tmp_file.name
            
            try:
                if file.name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(temp_path)
                elif file.name.lower().endswith('.docx'):
                    loader = Docx2txtLoader(temp_path)
                elif file.name.lower().endswith('.txt'):
                    loader = TextLoader(temp_path)
                else:
                    st.error(f"Unsupported file type: {file.name}")
                    continue
                    
                docs = loader.load()
                # Set the original filename in metadata
                for doc in docs:
                    doc.metadata['source'] = file.name  # Use original filename instead of temp file
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error loading {file.name}: {str(e)}")
            finally:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
    
    return documents

@log_timing
def process_documents(documents):
    if not documents:
        return None
        
    # Analyze documents
    stats = analyze_document_content(documents)
    st.session_state.document_stats = stats
    st.session_state.key_insights = generate_document_insights(stats)
    
    # Split documents into smaller chunks for faster processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Clear existing ChromaDB directory and recreate it
    PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
    if os.path.exists(PERSIST_DIR):
        try:
            # First, close any existing vector store
            if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
                try:
                    st.session_state.vector_store.persist()
                except:
                    pass
                st.session_state.vector_store = None
            
            # Wait a bit for resources to be released
            time.sleep(1)
            
            # Now try to remove the directory
            for retry in range(3):  # Try up to 3 times
                try:
                    shutil.rmtree(PERSIST_DIR)
                    break
                except Exception as e:
                    if retry == 2:  # Last attempt
                        st.warning(f"Could not clear old database: {str(e)}")
                    time.sleep(1)
            
            # Ensure the directory exists
            os.makedirs(PERSIST_DIR, exist_ok=True)
            
        except Exception as e:
            st.error(f"Error managing database directory: {str(e)}")
            if not os.path.exists(PERSIST_DIR):
                os.makedirs(PERSIST_DIR)
    
    # Create vector store with GPU-optimized embeddings
    try:
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=st.session_state.embeddings,
            persist_directory=PERSIST_DIR
        )
        vector_store.persist()
        
        # Get unique file names from documents
        processed_files = list(set(doc.metadata.get('source', '') for doc in documents))
        st.session_state.processed_files = processed_files
        
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

@log_timing
def get_qa_chain(vector_store):
    template = """Use the following context to answer the question thoroughly and completely. If you don't know, say "I don't know."

Context: {context}

Question: {question}
Answer: Let me provide a complete response based on the context provided."""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        st.session_state.llm,
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 3
            }
        ),
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "verbose": False
        },
        return_source_documents=False
    )
    
    return qa_chain

# System monitoring functions (for debugging only, not displayed in UI)
def log_system_metrics():
    """Log system metrics to a file for debugging"""
    try:
        metrics = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'gpu_info': get_gpu_info(),
            'system_info': get_system_info(),
            'processing_times': st.session_state.processing_times[-5:] if st.session_state.processing_times else []
        }
        
        log_dir = "system_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"system_metrics_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, 'a') as f:
            f.write(f"{json.dumps(metrics)}\n")
    except Exception as e:
        print(f"Error logging metrics: {str(e)}")

def get_gpu_info():
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            info = {
                'id': gpu.id,
                'name': gpu.name,
                'load': f"{gpu.load*100:.1f}%",
                'memory_used': f"{gpu.memoryUsed}MB",
                'memory_total': f"{gpu.memoryTotal}MB",
                'temperature': f"{gpu.temperature}°C"
            }
            gpu_info.append(info)
        return gpu_info
    except Exception as e:
        return [{"error": str(e)}]

def get_system_info():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    return {
        'cpu_usage': f"{cpu_percent}%",
        'memory_used': f"{memory.percent}%",
        'memory_available': f"{memory.available/1024/1024/1024:.1f}GB"
    }

# Main Interface
st.title("🤖 Advanced Document Q&A with RAG")

# Tabs for different features
tab1, tab2, tab3 = st.tabs(["Document Processing", "Analytics", "Chat History"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Clear previous vector store from session state
                if 'vector_store' in st.session_state:
                    if st.session_state.vector_store is not None:
                        try:
                            st.session_state.vector_store.persist()
                        except:
                            pass
                    st.session_state.vector_store = None
                
                documents = load_documents(uploaded_files)
                if documents:
                    st.session_state.vector_store = process_documents(documents)
                    if st.session_state.vector_store:
                        st.success("Documents processed successfully!")
                        
                        # Show document insights
                        st.subheader("📊 Document Insights")
                        for insight in st.session_state.key_insights:
                            st.info(insight)
                        
                        # Show processed files
                        if 'processed_files' in st.session_state:
                            st.write("📑 Processed files:")
                            for file in st.session_state.processed_files:
                                st.write(f"- {file}")
                    else:
                        st.error("Error processing documents.")
                else:
                    st.error("No documents were successfully processed.")

    # Q&A Interface with enhanced features
    if st.session_state.vector_store is not None:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'question' not in st.session_state:
                st.session_state.question = ""
            question = st.text_input("Ask a question about your documents:", value=st.session_state.question)
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        qa_chain = get_qa_chain(st.session_state.vector_store)
                        start_time = time.time()
                        response = qa_chain.invoke(question)
                        end_time = time.time()
                        
                        # Store in chat history
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': response['result'],
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'response_time': f"{end_time - start_time:.2f}s"
                        })
                        
                        st.write("### Answer:")
                        st.write(response['result'])
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
        
        with col2:
            st.write("💡 Question Suggestions:")
            suggestions = [
                "What are the main topics covered?",
                "Can you summarize the key points?",
                "What are the conclusions?",
                "Any specific recommendations?"
            ]
            for suggestion in suggestions:
                if st.button(suggestion, key=suggestion):
                    st.session_state.question = suggestion
                    question = suggestion

with tab2:
    if st.session_state.document_stats:
        st.subheader("📈 Document Analytics")
        
        # Sentence Length Distribution
        sentence_lengths = st.session_state.document_stats['sentence_lengths']
        df = pd.DataFrame({'Sentence Length': sentence_lengths})
        fig = px.histogram(df, x='Sentence Length', 
                          title='Sentence Length Distribution',
                          labels={'count': 'Number of Sentences'})
        st.plotly_chart(fig)
        
        # Key Phrases
        st.subheader("🔑 Key Phrases")
        phrases_df = pd.DataFrame(st.session_state.document_stats['key_phrases'], columns=['Phrase'])
        st.dataframe(phrases_df)
        
        # Document Statistics
        st.subheader("📊 Document Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Characters", st.session_state.document_stats['total_chars'])
        with col2:
            st.metric("Total Sentences", st.session_state.document_stats['total_sentences'])

with tab3:
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question']} ({chat['timestamp']})"):
                st.write("A:", chat['answer'])
                st.caption(f"Response time: {chat['response_time']}")
    else:
        st.info("No chat history yet. Start asking questions!")

# Remove sidebar system monitoring
with st.sidebar:
    st.markdown("""
    ### 🚀 Features
    - Document Analytics
    - Interactive Q&A
    - Chat History
    - Key Insights Generation
    - Smart Document Processing
    """)

# Log system metrics in the background (for debugging)
if 'last_metric_log' not in st.session_state:
    st.session_state.last_metric_log = time.time()

# Log metrics every 5 minutes
current_time = time.time()
if current_time - st.session_state.last_metric_log >= 300:  # 300 seconds = 5 minutes
    log_system_metrics()
    st.session_state.last_metric_log = current_time
