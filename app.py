# -*- coding: utf-8 -*-
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import shutil

# Suppress Streamlit torch warnings
import os
import sys

# Set environment variables before any torch-related imports
os.environ["PYTORCH_DISABLE_PER_OP_PROFILING"] = "1"
os.environ["TORCH_LOGS"] = ""
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
import warnings
import subprocess
import re

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*torch.*")

# Redirect torch warnings to suppress them completely
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents with OCR support for scanned PDFs"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            pdf_text = ""
            
            # First try normal text extraction
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
            
            # If no text found, try OCR with PyMuPDF
            if not pdf_text.strip():
                try:
                    import fitz  # PyMuPDF
                    import pytesseract
                    from PIL import Image
                    import io
                    
                    # Check if Tesseract is available
                    try:
                        # Set Tesseract path for Windows
                        import platform
                        if platform.system() == 'Windows':
                            possible_paths = [
                                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                                r"C:\Users\Eagle\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
                                r"C:\Users\Eagle\AppData\Local\Microsoft\WinGet\Packages\UB-Mannheim.TesseractOCR_Microsoft.Winget.Source_8wekyb3d8bbwe\tesseract.exe",
                                r"C:\tools\tesseract\tesseract.exe"
                            ]
                            tesseract_found = False
                            for path in possible_paths:
                                if os.path.exists(path):
                                    pytesseract.pytesseract.tesseract_cmd = path
                                    tesseract_found = True
                                    st.info(f"🔍 Found Tesseract at: {path}")
                                    break
                            
                            if not tesseract_found:
                                # Try to find tesseract in PATH
                                try:
                                    import shutil
                                    tesseract_path = shutil.which("tesseract")
                                    if tesseract_path:
                                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                                        tesseract_found = True
                                        st.info(f"🔍 Found Tesseract in PATH: {tesseract_path}")
                                except:
                                    pass
                        
                        # Test if Tesseract works
                        pytesseract.get_tesseract_version()
                        
                    except Exception as e:
                        st.error(f"❌ {pdf.name} is a scanned PDF but Tesseract OCR is not installed.")
                        st.info("🔧 **Install Tesseract OCR:**")
                        st.code("winget install UB-Mannheim.TesseractOCR")
                        st.info("Or download from: https://github.com/UB-Mannheim/tesseract/wiki")
                        st.info("After installation, restart the app.")
                        continue
                    
                    # Reset file pointer
                    pdf.seek(0)
                    pdf_bytes = pdf.read()
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                    
                    st.info(f"🔍 Using OCR for {pdf.name} (scanned document)")
                    
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        # Convert page to image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Perform OCR
                        try:
                            ocr_text = pytesseract.image_to_string(image, lang='eng')
                            if ocr_text.strip():
                                pdf_text += ocr_text + "\n"
                        except Exception as ocr_error:
                            st.warning(f"⚠️ OCR failed for page {page_num + 1}: {str(ocr_error)}")
                    
                    pdf_document.close()
                    
                except ImportError:
                    st.error(f"❌ {pdf.name} appears to be a scanned PDF but OCR libraries are missing.")
                    st.info("Run: pip install pymupdf pytesseract pillow")
                    continue
                except Exception as e:
                    st.error(f"❌ OCR failed for {pdf.name}: {str(e)}")
                    continue
            
            if pdf_text.strip():
                text += pdf_text
                st.success(f"✅ Processed {pdf.name}")
            else:
                st.error(f"❌ No text found in {pdf.name}")
                
        except Exception as e:
            st.error(f"❌ Error reading {pdf.name}: {str(e)}")
    
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create vector store from text chunks using torch-free TF-IDF embeddings"""
    try:
        # Use sklearn TF-IDF to completely avoid torch
        from sklearn.feature_extraction.text import TfidfVectorizer
        from langchain.embeddings.base import Embeddings
        import numpy as np
        
        class TorchFreeTFIDFEmbeddings(Embeddings):
            def __init__(self):
                self.vectorizer = TfidfVectorizer(
                    max_features=512, 
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                self.fitted = False
                self.feature_dim = 512
                
            def embed_documents(self, texts):
                if not self.fitted:
                    vectors = self.vectorizer.fit_transform(texts)
                    self.fitted = True
                else:
                    vectors = self.vectorizer.transform(texts)
                
                # Convert sparse matrix to dense and then to list
                dense_vectors = vectors.toarray()
                # Pad or truncate to fixed dimension
                result = []
                for vector in dense_vectors:
                    if len(vector) < self.feature_dim:
                        padded = np.pad(vector, (0, self.feature_dim - len(vector)), 'constant')
                    else:
                        padded = vector[:self.feature_dim]
                    result.append(padded.tolist())
                return result
                
            def embed_query(self, text):
                if not self.fitted:
                    return [0.0] * self.feature_dim
                    
                vector = self.vectorizer.transform([text])
                dense_vector = vector.toarray()[0]
                
                # Pad or truncate to fixed dimension
                if len(dense_vector) < self.feature_dim:
                    padded = np.pad(dense_vector, (0, self.feature_dim - len(dense_vector)), 'constant')
                else:
                    padded = dense_vector[:self.feature_dim]
                    
                return padded.tolist()
        
        st.info("🔧 Using TF-IDF embeddings (100% torch-free)")
        embeddings = TorchFreeTFIDFEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error with TF-IDF embeddings: {e}")
        
        # If even TF-IDF fails, use a simple fallback
        try:
            st.warning("🔄 Using basic text matching as fallback...")
            
            class SimpleFallbackEmbeddings(Embeddings):
                def __init__(self):
                    self.feature_dim = 300
                    
                def embed_documents(self, texts):
                    # Very simple hash-based embeddings
                    embeddings = []
                    for text in texts:
                        # Create a simple embedding based on text properties
                        words = text.lower().split()
                        embedding = [0.0] * self.feature_dim
                        
                        for i, word in enumerate(words[:self.feature_dim]):
                            embedding[i % self.feature_dim] += hash(word) % 1000 / 1000.0
                            
                        embeddings.append(embedding)
                    return embeddings
                    
                def embed_query(self, text):
                    words = text.lower().split()
                    embedding = [0.0] * self.feature_dim
                    
                    for i, word in enumerate(words[:self.feature_dim]):
                        embedding[i % self.feature_dim] += hash(word) % 1000 / 1000.0
                        
                    return embedding
            
            embeddings = SimpleFallbackEmbeddings()
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            st.warning("⚠️ Using basic text matching (limited accuracy)")
            return vectorstore
            
        except Exception as e2:
            st.error(f"❌ All embedding methods failed: {e2}")
            return None

def get_conversation_chain(vectorstore):
    """Create conversation chain"""
    # Get available models
    available_models = get_available_ollama_models()
    if not available_models:
        st.error("❌ No Ollama models found. Please install a model first.")
        st.info("Run: ollama pull llama3.2:1b")
        return None
    
    # Use the first available model
    model_name = available_models[0]
    
    try:
        llm = OllamaLLM(
            model=model_name,
            temperature=0.1
        )
        
        prompt_template = """You are a helpful assistant that answers questions based on the provided documents.
        
CONTEXT: {context}
QUESTION: {question}

ANSWER:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        retriever = vectorstore.as_retriever()

        # Use the updated ConversationalRetrievalChain without deprecated memory
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        st.session_state.model_info = {'model': model_name}
        return conversation_chain
        
    except Exception as e:
        st.error(f"❌ Error creating conversation chain: {e}")
        return None

def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        return []
    except:
        return []

def check_ollama_installation():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def handle_user_input(user_question):
    """Handle user input and generate response"""
    try:
        with st.spinner("🔍 Searching documents..."):
            # Get existing chat history from session state
            chat_history = st.session_state.get('chat_history', [])
            
            # Convert string format to tuples for LangChain compatibility
            formatted_history = []
            for i in range(0, len(chat_history), 2):
                if i + 1 < len(chat_history):
                    human_msg = chat_history[i].replace("Human: ", "")
                    ai_msg = chat_history[i + 1].replace("Assistant: ", "")
                    formatted_history.append((human_msg, ai_msg))
            
            # Invoke the conversation chain with question and formatted chat history
            response = st.session_state.conversation.invoke({
                'question': user_question,
                'chat_history': formatted_history
            })
        
        # Update chat history manually
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Add question and answer to history
        st.session_state.chat_history.append(f"Human: {user_question}")
        st.session_state.chat_history.append(f"Assistant: {response['answer']}")
                            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def main():
    load_dotenv()

    st.set_page_config(
        page_title="PDF Chat",
        page_icon="💬",
        layout="centered"
    )
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Simple header
    st.title("💬 PDF Chat")
    st.caption("Upload your documents and ask questions")
    
    # Check Ollama installation
    if not check_ollama_installation():
        st.error("❌ Ollama not detected. Please install Ollama first.")
        st.info("Download from: https://ollama.ai")
        if st.button("🔄 Check Again"):
            st.rerun()
        st.stop()
    
    # Show AI status if available
    if hasattr(st.session_state, 'model_info') and st.session_state.model_info:
        model_name = st.session_state.model_info.get('model', 'AI Model')
        st.success(f"✅ {model_name} ready")
    
    # File upload section
    st.subheader("📄 Upload Documents")
    pdf_docs = st.file_uploader(
        "Choose PDF files",
        accept_multiple_files=True,
        type="pdf"
    )
    
    if st.button("🔄 Process Documents"):
        if not pdf_docs:
            st.warning("⚠️ Please upload at least one PDF file.")
        else:
            with st.spinner("📄 Processing documents..."):
                try:
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("❌ No text found in the uploaded PDFs.")
                    else:
                        # Create chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.info(f"✂️ Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)
                        if not vectorstore:
                            st.error("❌ Failed to create knowledge base")
                            return
                        st.info("🧠 Created knowledge base")

                        # Create conversation chain
                        conversation_chain = get_conversation_chain(vectorstore)
                        
                        if conversation_chain:
                            st.session_state.conversation = conversation_chain
                            st.success("✅ Ready to chat!")
                        else:
                            st.error("❌ Failed to set up AI system.")
                            
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")

    # Chat section
    st.subheader("💬 Chat")
    
    # Display existing chat history first
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        # Display conversation in reverse order (most recent first)
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            message = st.session_state.chat_history[i]
            if i % 2 == 0:  # User messages (even indices)
                user_msg = message.replace("Human: ", "")
                st.write(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
            else:  # Bot messages (odd indices)
                bot_msg = message.replace("Assistant: ", "")
                st.write(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)
    
    # User input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="Type your question here..."
        )
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_question:
        if st.session_state.conversation:
            handle_user_input(user_question)
            st.rerun()  # Refresh to show the new conversation
        else:
            st.warning("📁 Please upload and process your documents first!")

if __name__ == '__main__':
    main()
