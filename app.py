import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs, ocr_language='eng'):
    import pytesseract
    from PIL import Image
    import io
    import fitz  # PyMuPDF as alternative to pdf2image
    import os
    
    # Set Tesseract path for Windows
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\Eagle\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        "tesseract"  # If it's in PATH
    ]
    
    tesseract_found = False
    for path in possible_paths:
        if os.path.exists(path) or path == "tesseract":
            try:
                pytesseract.pytesseract.tesseract_cmd = path
                # Test if tesseract works
                pytesseract.get_tesseract_version()
                tesseract_found = True
                break
            except:
                continue
    
    text = ""
    
    if not pdf_docs:
        return text

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            pdf_text = ""
            
            # Reset file pointer for PyMuPDF processing
            pdf.seek(0)
            pdf_bytes = pdf.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Process each page
            for page_num in range(len(pdf_document)):
                # First, try to extract text normally with PyPDF
                if page_num < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text and page_text.strip():
                        pdf_text += page_text + "\n"
                        continue
                
                # Try PyMuPDF text extraction as first fallback
                fitz_page = pdf_document[page_num]
                fitz_text = fitz_page.get_text()
                if fitz_text and fitz_text.strip():
                    pdf_text += fitz_text + "\n"
                    st.info(f"✓ PyMuPDF extracted text from page {page_num + 1}")
                    continue
                
                # If still no text found, try OCR on this page (only if Tesseract is available)
                if tesseract_found:
                    st.info(f"Using OCR for page {page_num + 1} of {pdf.name}...")
                    try:
                        # Convert page to image with higher resolution
                        pix = fitz_page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x zoom for better OCR
                        img_data = pix.tobytes("png")
                        
                        # Convert to PIL Image
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Perform OCR on the image
                        ocr_text = pytesseract.image_to_string(image, lang=ocr_language)
                        if ocr_text.strip():
                            pdf_text += ocr_text + "\n"
                            st.success(f"✓ OCR extracted text from page {page_num + 1}")
                        else:
                            st.warning(f"⚠️ No text found on page {page_num + 1}")
                            
                    except Exception as ocr_error:
                        st.warning(f"OCR failed for page {page_num + 1}: {str(ocr_error)}")
                        continue
                else:
                    st.warning(f"⚠️ Tesseract not available - skipping OCR for page {page_num + 1}")
                    st.info("💡 Install Tesseract: `winget install UB-Mannheim.TesseractOCR`")
            
            pdf_document.close()
            
            if pdf_text.strip():
                text += pdf_text
                st.info(f"✓ Successfully processed {pdf.name}")
            else:
                st.warning(f"⚠️ No text could be extracted from {pdf.name}")
                
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            continue

    return text


def get_text_chunks(raw_text):
    # Use a more sophisticated splitter that considers tokens
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Conservative chunk size for small models
        chunk_overlap=30,  # Small overlap
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try different separators
    )

    chunks = text_splitter.split_text(raw_text)
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20]

    return chunks 


def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks provided. Please ensure your PDFs contain readable text.")
        
    # Using the new HuggingFaceEmbeddings class with a smaller, faster model
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}  # Ensure CPU usage for compatibility
    )
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):
    # Use HuggingFace Pipeline for local inference (no API token needed)
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline
    
    try:
        # Create a local text generation pipeline
        st.info("Loading local language model...")
        text_generation_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",  # Using smaller model for faster local inference
            max_new_tokens=150,  # Limit output tokens
            do_sample=True,
            temperature=0.7,
            truncation=True  # Truncate long inputs
        )
        
        llm = HuggingFacePipeline(
            pipeline=text_generation_pipeline,
            model_kwargs={
                "temperature": 0.7, 
                "max_new_tokens": 150,
                "truncation": True
            }
        )
        
        st.success("Local model loaded successfully!")
        
    except Exception as e:
        st.error(f"Failed to load local model: {e}")
        # Simple fallback using a basic text generator
        st.warning("Using a simple text completion model as fallback...")
        
        try:
            simple_pipeline = pipeline(
                "text-generation",
                model="gpt2",  # Small, widely available model
                max_new_tokens=80,  # Reduced token limit
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256,
                truncation=True
            )
            
            llm = HuggingFacePipeline(
                pipeline=simple_pipeline,
                model_kwargs={
                    "max_new_tokens": 80, 
                    "temperature": 0.7,
                    "truncation": True
                }
            )
            st.info("Fallback model loaded successfully!")
            
        except Exception as e2:
            st.error(f"All models failed to load: {e2}")
            st.error("Please check your internet connection and try again.")
            return None
    
    from langchain.memory import ConversationBufferMemory
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'  # Specify the output key for better compatibility
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False  # Simplify the response
    )

    return conversation_chain


def handle_user_input(user_question):
    try:
        # Use the newer invoke method instead of deprecated __call__
        response = st.session_state.conversation.invoke({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Clear the chat area and display messages in proper chronological order
        # The chat_history alternates: [user1, bot1, user2, bot2, ...]
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                # User messages (even indices) - questions
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                # Bot messages (odd indices) - answers
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.error("Please try asking a different question or reprocess your documents.")
        
            

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    
    # Add info about supported PDF types
    with st.expander("ℹ️ Supported PDF Types"):
        st.markdown("""
        **This app supports:**
        - ✅ **Text-based PDFs** - Regular PDFs with selectable text
        - ✅ **Scanned PDFs** - Image-based PDFs (uses OCR)
        - ✅ **Mixed PDFs** - PDFs with both text and scanned pages
        
        **Note:** OCR processing may take longer for scanned documents.
        """)
    
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        if st.session_state.conversation is not None:
            handle_user_input(user_question)
        else:
            st.warning("Please upload and process PDF documents first!")

    st.write(user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        
        # OCR Configuration
        with st.expander("🔍 OCR Settings (for scanned PDFs)"):
            st.info("OCR will automatically activate for pages without readable text")
            st.markdown("**Uses PyMuPDF + Tesseract (no Poppler required)**")
            
            # Check if Tesseract is available
            try:
                import pytesseract
                import os
                
                # Set Tesseract path for Windows
                possible_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r"C:\Users\Eagle\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
                    "tesseract"  # If it's in PATH
                ]
                
                tesseract_found = False
                tesseract_path = None
                for path in possible_paths:
                    if os.path.exists(path) or path == "tesseract":
                        try:
                            pytesseract.pytesseract.tesseract_cmd = path
                            tesseract_version = pytesseract.get_tesseract_version()
                            tesseract_found = True
                            tesseract_path = path
                            break
                        except:
                            continue
                
                if tesseract_found:
                    st.success(f"✅ Tesseract available at: {tesseract_path}")
                else:
                    raise Exception("Tesseract not found")
                    
            except Exception as e:
                st.warning("⚠️ Tesseract not found or not accessible")
                st.markdown("""
                **To enable OCR, install Tesseract:**
                - **Windows**: `winget install UB-Mannheim.TesseractOCR`
                - **Or download**: https://github.com/UB-Mannheim/tesseract/wiki
                - **Restart the app** after installation
                """)
                st.info("The app will still work for text-based PDFs without OCR")
            
            ocr_language = st.selectbox(
                "OCR Language:",
                ["eng", "fra", "deu", "spa", "ita", "por"],
                index=1 if st.session_state.get('ocr_language') == 'fra' else 0,
                help="Select the primary language in your documents"
            )
            st.session_state.ocr_language = ocr_language
        
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            help="Supports both text-based and scanned/image PDFs"
        )
        
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing documents... This may take a few minutes on first run."):
                    try:
                        # Get PDF text
                        st.info("📄 Extracting text from PDFs...")
                        ocr_lang = getattr(st.session_state, 'ocr_language', 'eng')
                        raw_text = get_pdf_text(pdf_docs, ocr_lang)
                        
                        if not raw_text.strip():
                            st.error("No readable text found in the uploaded PDFs. Please ensure your PDFs contain text (not just images).")
                        else:
                            # Get the text chunks
                            st.info("✂️ Splitting text into chunks...")
                            text_chunks = get_text_chunks(raw_text)
                            
                            if not text_chunks:
                                st.error("Could not create text chunks. Please try uploading different PDFs.")
                            else:
                                # Create vector store
                                st.info("🤖 Creating embeddings (downloading model if first time)...")
                                vectorstore = get_vectorstore(text_chunks)

                                # Create conversation chain
                                st.info("🔗 Setting up conversation chain...")
                                conversation_chain = get_conversation_chain(vectorstore)
                                
                                if conversation_chain is not None:
                                    st.session_state.conversation = conversation_chain
                                    st.success("✅ Documents processed successfully! You can now ask questions.")
                                else:
                                    st.error("Failed to create conversation chain. Please try again.")
                                
                    except Exception as e:
                        st.error(f"An error occurred while processing: {str(e)}")
                        st.error("Please try again with different PDF files.")

if __name__ == '__main__':
    main()