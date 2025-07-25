import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import warnings
import os
import asyncio

# Suppress common warnings that don't affect functionality
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*Examining the path.*")

# Set environment variable to reduce torch verbosity
# Removed invalid TORCH_LOGS setting


def get_pdf_text(pdf_docs, ocr_language='eng'):
    import pytesseract
    from PIL import Image
    import io
    import fitz  # PyMuPDF as alternative to pdf2image
    import os
    import re
    
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
    document_metadata = {"title": "", "sections": [], "page_count": 0}
    
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
            
            document_metadata["page_count"] = len(pdf_document)
            
            # Process each page with enhanced structure recognition
            for page_num in range(len(pdf_document)):
                page_header = f"\n\n--- PAGE {page_num + 1} ---\n"
                
                # First, try to extract text normally with PyPDF
                if page_num < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text and page_text.strip():
                        # Enhanced text processing
                        processed_text = enhance_text_structure(page_text, page_num)
                        pdf_text += page_header + processed_text + "\n"
                        
                        # Extract title from first page
                        if page_num == 0 and not document_metadata["title"]:
                            document_metadata["title"] = extract_document_title(page_text)
                        
                        # Extract sections
                        sections = extract_sections(page_text)
                        document_metadata["sections"].extend(sections)
                        continue
                
                # Try PyMuPDF text extraction as first fallback
                fitz_page = pdf_document[page_num]
                fitz_text = fitz_page.get_text()
                if fitz_text and fitz_text.strip():
                    processed_text = enhance_text_structure(fitz_text, page_num)
                    pdf_text += page_header + processed_text + "\n"
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
                            processed_text = enhance_text_structure(ocr_text, page_num)
                            pdf_text += page_header + processed_text + "\n"
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
                # Add document metadata to the beginning
                metadata_header = f"""DOCUMENT: {pdf.name}
TITLE: {document_metadata.get('title', 'Unknown')}
PAGES: {document_metadata['page_count']}
SECTIONS: {len(document_metadata['sections'])}

"""
                text += metadata_header + pdf_text
                st.info(f"✓ Successfully processed {pdf.name} with enhanced structure")
            else:
                st.warning(f"⚠️ No text could be extracted from {pdf.name}")
                
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            continue

    return text


def enhance_text_structure(text, page_num):
    """Enhance text with better structure recognition"""
    import re
    
    # Clean up common OCR/extraction issues
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
    
    # Identify and format headers (simple heuristic)
    lines = text.split('\n')
    enhanced_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            enhanced_lines.append('')
            continue
            
        # Detect potential headers (short lines, title case, etc.)
        if (len(line) < 100 and 
            (line.isupper() or line.istitle()) and 
            not line.endswith('.') and
            len(line.split()) < 10):
            enhanced_lines.append(f"\n## {line}\n")
        else:
            enhanced_lines.append(line)
    
    return '\n'.join(enhanced_lines)


def extract_document_title(text):
    """Extract document title from first page"""
    import re
    lines = text.split('\n')[:10]  # Check first 10 lines
    
    for line in lines:
        line = line.strip()
        if (len(line) > 10 and len(line) < 200 and 
            not line.startswith('Page') and
            not re.match(r'^\d+$', line)):
            return line
    return "Document"


def extract_sections(text):
    """Extract section headers from text"""
    import re
    sections = []
    
    # Simple section detection
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if (line and len(line) < 100 and 
            (line.isupper() or line.istitle()) and
            not line.endswith('.') and
            len(line.split()) >= 2 and len(line.split()) <= 8):
            sections.append(line)
    
    return sections


def get_text_chunks(raw_text):
    """Advanced text chunking with context preservation and quality assessment"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import re
    
    # Smart chunk size adjustment based on document length
    doc_length = len(raw_text)
    if doc_length < 10000:
        chunk_size = 600
        chunk_overlap = 150
    elif doc_length < 50000:
        chunk_size = 800
        chunk_overlap = 200
    else:
        chunk_size = 1000
        chunk_overlap = 250
    
    # Enhanced separators that respect document structure
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n## ",      # Section headers
            "\n### ",     # Subsection headers
            "\n--- PAGE", # Page breaks
            "\n\n\n",     # Large gaps
            "\n\n",       # Paragraph breaks
            "\n",         # Line breaks
            ". ",         # Sentence endings
            "! ",         # Exclamation endings
            "? ",         # Question endings
            "; ",         # Semicolons
            ": ",         # Colons
            " ",          # Spaces
            ""            # Character level
        ]
    )

    chunks = text_splitter.split_text(raw_text)
    
    # Enhanced chunk processing and quality assessment
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        
        # Quality checks
        if len(chunk) < 100:  # Too short
            continue
            
        if len(chunk.split()) < 10:  # Too few words
            continue
            
        # Calculate chunk quality score
        quality_score = assess_chunk_quality(chunk)
        
        if quality_score > 0.3:  # Minimum quality threshold
            # Add contextual metadata
            chunk_metadata = {
                "chunk_id": i,
                "length": len(chunk),
                "word_count": len(chunk.split()),
                "quality_score": quality_score,
                "has_sections": "##" in chunk,
                "page_info": extract_page_info(chunk)
            }
            
            # Enhanced chunk with context markers
            enhanced_chunk = add_context_markers(chunk, chunk_metadata)
            enhanced_chunks.append(enhanced_chunk)
    
    st.info(f"✂️ Created {len(enhanced_chunks)} high-quality chunks from {len(chunks)} raw chunks")
    return enhanced_chunks


def assess_chunk_quality(chunk):
    """Assess the quality of a text chunk"""
    import re
    
    score = 0.0
    
    # Length factors
    if 200 <= len(chunk) <= 1500:
        score += 0.3
    elif 100 <= len(chunk) <= 2000:
        score += 0.2
    
    # Content quality
    word_count = len(chunk.split())
    if word_count >= 20:
        score += 0.2
    
    # Sentence structure
    sentences = re.split(r'[.!?]+', chunk)
    if len(sentences) >= 3:
        score += 0.2
    
    # Information density (not just repeated chars)
    unique_words = len(set(chunk.lower().split()))
    if unique_words / max(word_count, 1) > 0.6:
        score += 0.2
    
    # Has meaningful content (not just numbers/symbols)
    meaningful_chars = re.findall(r'[a-zA-Z]', chunk)
    if len(meaningful_chars) / len(chunk) > 0.7:
        score += 0.1
    
    return min(score, 1.0)


def extract_page_info(chunk):
    """Extract page information from chunk"""
    import re
    page_match = re.search(r'--- PAGE (\d+) ---', chunk)
    return int(page_match.group(1)) if page_match else None


def add_context_markers(chunk, metadata):
    """Add contextual information to chunks"""
    context_header = ""
    
    if metadata.get("page_info"):
        context_header += f"[Page {metadata['page_info']}] "
    
    if metadata.get("has_sections"):
        context_header += "[Contains Sections] "
    
    if metadata.get("quality_score", 0) > 0.8:
        context_header += "[High Quality] "
    
    return context_header + chunk 


def get_vectorstore(text_chunks):
    """Create enhanced vector store with hybrid retrieval capabilities"""
    if not text_chunks:
        raise ValueError("No text chunks provided. Please ensure your PDFs contain readable text.")
    
    # Store chunks for BM25 retrieval
    st.session_state.text_chunks = text_chunks
    
    try:
        # Using enhanced embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create enhanced metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(text_chunks):
            metadata = {
                "chunk_id": i,
                "length": len(chunk),
                "word_count": len(chunk.split()),
                "quality_score": assess_chunk_quality(chunk),
                "source": f"chunk_{i}"
            }
            metadatas.append(metadata)
        
        # Create vector store with enhanced metadata
        vectorstore = FAISS.from_texts(
            text_chunks,
            embeddings,
            metadatas=metadatas
        )
        
        st.success(f"✅ Created vector store with {len(text_chunks)} chunks")
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise


def create_hybrid_retriever(vectorstore, text_chunks, k=8):
    """Create hybrid retriever combining BM25 and vector search"""
    try:
        from rank_bm25 import BM25Okapi
        import numpy as np
        
        # Prepare BM25
        tokenized_chunks = [chunk.lower().split() for chunk in text_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
        def hybrid_search(query, k=k):
            """Perform hybrid search combining BM25 and vector similarity"""
            
            # BM25 search (keyword-based)
            query_tokens = query.lower().split()
            bm25_scores = bm25.get_scores(query_tokens)
            
            # Get top BM25 results
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:k*2]
            
            # Vector search (semantic)
            vector_results = vectorstore.similarity_search_with_score(query, k=k*2)
            
            # Combine and rank results
            combined_results = {}
            
            # Add BM25 results with scores
            for i, idx in enumerate(bm25_top_indices):
                if bm25_scores[idx] > 0:  # Only include relevant BM25 results
                    combined_results[idx] = {
                        'chunk': text_chunks[idx],
                        'bm25_score': bm25_scores[idx],
                        'bm25_rank': i,
                        'vector_score': 0,
                        'vector_rank': k*2
                    }
            
            # Add vector results with scores
            for i, (doc, score) in enumerate(vector_results):
                chunk_idx = text_chunks.index(doc.page_content)
                if chunk_idx in combined_results:
                    combined_results[chunk_idx]['vector_score'] = 1 - score  # Convert distance to similarity
                    combined_results[chunk_idx]['vector_rank'] = i
                else:
                    combined_results[chunk_idx] = {
                        'chunk': doc.page_content,
                        'bm25_score': 0,
                        'bm25_rank': k*2,
                        'vector_score': 1 - score,
                        'vector_rank': i
                    }
            
            # Calculate hybrid scores (weighted combination)
            for idx in combined_results:
                result = combined_results[idx]
                # Normalize scores
                bm25_norm = result['bm25_score'] / (max(bm25_scores) + 1e-6)
                vector_norm = result['vector_score']
                
                # Weighted combination (favor semantic but include keyword matching)
                result['hybrid_score'] = 0.3 * bm25_norm + 0.7 * vector_norm
            
            # Sort by hybrid score and return top k
            sorted_results = sorted(
                combined_results.items(),
                key=lambda x: x[1]['hybrid_score'],
                reverse=True
            )[:k]
            
            return [result[1]['chunk'] for result in sorted_results]
        
        return hybrid_search
        
    except ImportError:
        st.warning("⚠️ rank-bm25 not installed. Using vector search only.")
        return lambda query, k=k: [doc.page_content for doc in vectorstore.similarity_search(query, k=k)]
    except Exception as e:
        st.warning(f"Hybrid search setup failed: {e}. Using vector search only.")
        return lambda query, k=k: [doc.page_content for doc in vectorstore.similarity_search(query, k=k)]


def get_conversation_chain(vectorstore, model_type, model_version):
    """Create conversation chain based on selected model type and version."""
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline

    llm = None
    model_info = {}

    if model_type == "HuggingFace FLAN-T5":
        try:
            st.info(f"🧠 Loading HuggingFace model: {model_version}...")
            text_generation_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",  # Updated to a free model
                max_new_tokens=300,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                truncation=True
            )

            llm = HuggingFacePipeline(
                pipeline=text_generation_pipeline,
                model_kwargs={
                    "temperature": 0.2,
                    "max_new_tokens": 300,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "truncation": True
                }
            )

            model_info = {
                'provider': 'HuggingFace',
                'model': "google/flan-t5-base",  # Updated to a free model
                'temperature': 0.2,
                'max_tokens': 300,
                'status': 'active'
            }

            st.success(f"🎯 HuggingFace {model_version} model loaded successfully!")

        except Exception as e:
            st.error(f"Failed to load HuggingFace model: {e}")
            return None

    # Store model info in session state
    st.session_state.model_info = model_info

    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate

    # Enhanced prompt template optimized for HuggingFace models
    prompt_template = """You are an expert document analyst. Provide accurate answers based on the document context provided.

CONTEXT: {context}

QUESTION: {question}

INSTRUCTIONS:
- Base your answer on the provided context
- Be specific and detailed when possible
- Use clear, professional language
- If information is missing, state this clearly
- Include relevant details from the documents

ANSWER:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Enhanced memory system
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'  # Specify which output to store in memory
    )

    # Use the vectorstore directly as a retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Create conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        verbose=False,
        max_tokens_limit=3000
    )

    return conversation_chain


def generate_query_variations(original_query):
    """Generate variations of the user query for better retrieval coverage"""
    variations = [original_query]
    
    # Simple query expansion techniques
    query_lower = original_query.lower()
    
    # Add question variations
    if not query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
        variations.append(f"What is {original_query}?")
        variations.append(f"Explain {original_query}")
    
    # Add keyword extraction
    words = original_query.split()
    if len(words) > 2:
        # Create shorter keyword queries
        important_words = [w for w in words if len(w) > 3 and w.lower() not in 
                          ['the', 'and', 'but', 'for', 'with', 'about', 'this', 'that']]
        if important_words:
            variations.append(' '.join(important_words[:3]))
    
    # Add context-specific variations
    if 'summary' in query_lower or 'summarize' in query_lower:
        variations.append("main points key findings overview")
    
    if 'conclusion' in query_lower or 'result' in query_lower:
        variations.append("conclusions results findings outcomes")
    
    return variations[:4]  # Limit to 4 variations


def filter_chunks_by_relevance(chunks, query):
    """Filter and rank chunks by relevance to query"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        if not chunks:
            return []
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        # Combine query and chunks for vectorization
        all_texts = [query] + chunks
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarities
        query_vector = tfidf_matrix[0:1]
        chunk_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        
        # Sort chunks by similarity
        chunk_scores = list(zip(chunks, similarities))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out very low relevance chunks
        filtered_chunks = [chunk for chunk, score in chunk_scores if score > 0.1]
        
        return filtered_chunks
        
    except ImportError:
        # Fallback: simple keyword matching
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            scored_chunks.append((chunk, overlap))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks if score > 0]
    
    except Exception:
        return chunks  # Return all chunks if filtering fails


def handle_user_input(user_question):
    """Enhanced user input handling with quality assessment and smart features"""
    try:
        # Pre-process and enhance the user question
        enhanced_question = enhance_user_question(user_question)
        
        # Always use the standard conversation chain invoke method
        with st.spinner("🔍 Performing advanced document search..."):
            response = st.session_state.conversation.invoke({'question': enhanced_question})
        
        st.session_state.chat_history = response['chat_history']

        # Display conversation with enhanced formatting
        display_enhanced_conversation(response, user_question)
                            
    except Exception as e:
        handle_error_with_suggestions(e, user_question)


def enhance_user_question(question):
    """Enhance user question for better processing"""
    import re
    
    # Clean up the question
    question = question.strip()
    
    # Add question mark if missing
    if not question.endswith(('?', '.', '!')):
        question += '?'
    
    # Expand common abbreviations and improve clarity
    enhancements = {
        'whats': 'what is',
        'hows': 'how is',
        'wheres': 'where is',
        'cant': 'cannot',
        'wont': 'will not',
        'dont': 'do not',
        'isnt': 'is not',
        'arent': 'are not'
    }
    
    for abbrev, full in enhancements.items():
        question = re.sub(r'\b' + abbrev + r'\b', full, question, flags=re.IGNORECASE)
    
    return question


def display_enhanced_conversation(response, original_question):
    """Display conversation with enhanced formatting and analysis"""
    
    # Display messages in chronological order with quality indicators
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # User messages
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # Bot messages with enhanced analysis
            answer_content = message.content
            
            # Assess answer quality
            quality_assessment = assess_answer_quality(answer_content, original_question)
            
            # Create enhanced answer display
            enhanced_answer = create_enhanced_answer_display(answer_content, quality_assessment)
            
            st.write(bot_template.replace("{{MSG}}", enhanced_answer), unsafe_allow_html=True)
            
            # Show detailed source information
            display_source_analysis(response)
            
            # Show answer insights
            display_answer_insights(quality_assessment, answer_content)


def assess_answer_quality(answer, question):
    """Assess the quality of generated answer"""
    assessment = {
        'length_score': 0,
        'relevance_score': 0,
        'completeness_score': 0,
        'confidence_level': 'Medium',
        'suggestions': []
    }
    
    # Length assessment
    if len(answer) > 200:
        assessment['length_score'] = 0.9
    elif len(answer) > 100:
        assessment['length_score'] = 0.7
    elif len(answer) > 50:
        assessment['length_score'] = 0.5
    else:
        assessment['length_score'] = 0.2
        assessment['suggestions'].append("Answer might be too brief")
    
    # Relevance assessment (simple keyword overlap)
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(question_words.intersection(answer_words))
    assessment['relevance_score'] = min(overlap / max(len(question_words), 1), 1.0)
    
    # Completeness indicators
    if "don't have enough information" in answer.lower():
        assessment['completeness_score'] = 0.3
        assessment['confidence_level'] = 'Low'
        assessment['suggestions'].append("Document may not contain this information")
    elif any(word in answer.lower() for word in ['specifically', 'details', 'according to']):
        assessment['completeness_score'] = 0.9
        assessment['confidence_level'] = 'High'
    else:
        assessment['completeness_score'] = 0.6
    
    # Overall confidence
    overall_score = (assessment['length_score'] + assessment['relevance_score'] + assessment['completeness_score']) / 3
    
    if overall_score > 0.8:
        assessment['confidence_level'] = 'Very High'
    elif overall_score > 0.6:
        assessment['confidence_level'] = 'High'
    elif overall_score > 0.4:
        assessment['confidence_level'] = 'Medium'
    else:
        assessment['confidence_level'] = 'Low'
    
    return assessment


def create_enhanced_answer_display(answer, quality_assessment):
    """Create enhanced answer display with quality indicators"""
    
    # Choose confidence emoji and text
    confidence_indicators = {
        'Very High': '🎯 Comprehensive Answer',
        'High': '✅ Detailed Response', 
        'Medium': '📝 Good Response',
        'Low': '⚠️ Limited Information'
    }
    
    confidence_text = confidence_indicators.get(quality_assessment['confidence_level'], '📝 Response')
    
    # Create formatted answer
    formatted_answer = f"**{confidence_text}**\n\n{answer}"
    
    # Add quality insights
    if quality_assessment['suggestions']:
        formatted_answer += f"\n\n💡 *Note: {'; '.join(quality_assessment['suggestions'])}*"
    
    return formatted_answer


def display_source_analysis(response):
    """Display enhanced source information and analysis"""
    if hasattr(response, 'source_documents') and response.get('source_documents'):
        with st.expander("📚 Source Analysis & References", expanded=False):
            
            st.write("**📊 Source Quality Analysis:**")
            
            sources = response['source_documents'][:4]  # Show top 4 sources
            
            for idx, doc in enumerate(sources):
                source_content = doc.page_content[:300]
                
                # Assess source quality
                source_quality = assess_chunk_quality(source_content)
                
                quality_bar = "🟢" * int(source_quality * 5) + "⚪" * (5 - int(source_quality * 5))
                
                st.write(f"**Source {idx + 1}** - Quality: {quality_bar} ({source_quality:.1f}/1.0)")
                
                # Show source preview
                with st.expander(f"📄 Preview Source {idx + 1}"):
                    st.write(f"```\n{source_content}...\n```")
                    
                    if hasattr(doc, 'metadata') and doc.metadata:
                        st.write("**Metadata:**", doc.metadata)


def display_answer_insights(quality_assessment, answer):
    """Display insights about the answer quality and suggestions"""
    
    if quality_assessment['confidence_level'] in ['Low', 'Medium']:
        with st.expander("💡 Answer Insights & Suggestions", expanded=False):
            
            st.write("**📈 Answer Quality Metrics:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Length", f"{quality_assessment['length_score']:.1f}/1.0")
            with col2:
                st.metric("Relevance", f"{quality_assessment['relevance_score']:.1f}/1.0")
            with col3:
                st.metric("Completeness", f"{quality_assessment['completeness_score']:.1f}/1.0")
            
            if quality_assessment['suggestions']:
                st.write("**🔧 Suggestions for Better Results:**")
                for suggestion in quality_assessment['suggestions']:
                    st.write(f"• {suggestion}")
            
            st.write("**💭 Try These Alternative Questions:**")
            st.write("• Can you provide more details about this topic?")
            st.write("• What are the key points mentioned in the document?")
            st.write("• Are there any specific examples or data mentioned?")


def handle_error_with_suggestions(error, question):
    """Enhanced error handling with helpful suggestions"""
    st.error(f"❌ Error generating response: {str(error)}")
    
    # Analyze the error and provide specific suggestions
    error_str = str(error).lower()
    
    if "token" in error_str or "length" in error_str:
        st.warning("🔧 **Token Limit Issue**: Try asking a more specific question or break your question into smaller parts.")
        
    elif "connection" in error_str or "network" in error_str:
        st.warning("🌐 **Connection Issue**: Check your internet connection and try again.")
        
    elif "memory" in error_str:
        st.warning("💾 **Memory Issue**: Try reprocessing your documents or restart the application.")
        
    else:
        st.warning("🔄 **General Error**: Try rephrasing your question or reprocess your documents.")
    
    # Provide helpful suggestions based on question type
    question_lower = question.lower()
    
    with st.expander("💡 **Troubleshooting Tips**", expanded=True):
        st.markdown("""
        **For better results, try:**
        - ✅ Ask specific questions about document content
        - ✅ Use keywords that likely appear in your documents  
        - ✅ Break complex questions into simpler parts
        - ✅ Ensure your PDFs were processed successfully
        - ✅ Try the suggested question buttons above
        """)
        
        if any(word in question_lower for word in ['summary', 'summarize', 'overview']):
            st.info("📋 **For summaries**: Try 'What are the main points?' or use the Summarize button")
            
        elif any(word in question_lower for word in ['how', 'method', 'process']):
            st.info("🔧 **For process questions**: Try 'What methodology is described?' or 'What steps are mentioned?'")
            
        elif any(word in question_lower for word in ['why', 'reason', 'because']):
            st.info("🤔 **For reasoning questions**: Try 'What conclusions are drawn?' or 'What evidence is provided?'")
        
            

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("🤖 Enhanced PDF Chat with AI Models")
    
    # Display current model information
    if hasattr(st.session_state, 'model_info') and st.session_state.model_info:
        model_info = st.session_state.model_info
        provider = model_info.get('provider', 'Unknown')
        model_name = model_info.get('model', 'Unknown')
        status = model_info.get('status', 'Unknown')
        
        if provider == 'OpenAI':
            st.success(f"🚀 **Active Model**: {provider} {model_name} | Status: {status.title()}")
        else:
            st.info(f"🤖 **Active Model**: {provider} {model_name} | Status: {status.title()}")
    
    # Add info about supported PDF types
    with st.expander("ℹ️ Supported PDF Types & AI Models"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📄 PDF Support:**
            - ✅ **Text-based PDFs** - Regular PDFs with selectable text
            - ✅ **Scanned PDFs** - Image-based PDFs (uses OCR)
            - ✅ **Mixed PDFs** - PDFs with both text and scanned pages
            
            **Note:** OCR processing may take longer for scanned documents.
            """)
        
        with col2:
            st.markdown("""
            **🤖 AI Models:**
            - 🚀 **OpenAI GPT** - Premium models (requires API key)
              - GPT-4, GPT-3.5-turbo
              - Superior reasoning and analysis
            - 🔄 **HuggingFace** - Free open-source models
              - FLAN-T5 (base/small)
              - No API key required
            """)
    
    # OpenAI Configuration Section
    with st.expander("⚙️ OpenAI Configuration & Troubleshooting", expanded=False):
        st.markdown("""
        **🔑 To use OpenAI models:**
        1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Create a `.env` file in your project folder
        3. Add: `OPENAI_API_KEY=your_api_key_here`
        4. Restart the application
        """)
        
        import os
        api_key = os.getenv('OPENAI_API_KEY', '')
        
        if api_key and not api_key.startswith('your_'):
            st.success("✅ OpenAI API key detected!")
            
            col1, col2 = st.columns(2)
            with col1:
                current_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
                st.info(f"📊 **Model**: {current_model}")
            with col2:
                current_temp = os.getenv('OPENAI_TEMPERATURE', '0.2')
                st.info(f"🌡️ **Temperature**: {current_temp}")
        else:
            st.warning("⚠️ No OpenAI API key found. Using free HuggingFace models.")
            
        st.markdown("""
        **💰 Cost Information:**
        - GPT-3.5-turbo: ~$0.001-0.002 per 1K tokens
        - GPT-4: ~$0.03-0.06 per 1K tokens
        - Typical conversation: $0.01-0.10 per question
        
        **🚨 Common Issues:**
        - **Quota Exceeded**: Check billing at https://platform.openai.com/account/billing
        - **Invalid API Key**: Verify key at https://platform.openai.com/api-keys
        - **Rate Limits**: Wait and try again, or upgrade your plan
        """)
    
    # Add suggested questions section
    if st.session_state.conversation is not None:
        with st.expander("💡 Suggested Questions", expanded=False):
            st.markdown("""
            **Try asking questions like:**
            - "What is the main topic/theme of this document?"
            - "Summarize the key findings or conclusions"
            - "What are the main arguments presented?"
            - "List the important facts or statistics mentioned"
            - "What recommendations are provided?"
            - "Who are the main authors or contributors?"
            - "What methodology was used in this research?"
            - "What are the limitations mentioned?"
            """)
            
            # Quick question buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📋 Summarize"):
                    st.session_state.suggested_question = "Please provide a comprehensive summary of the main points in this document."
            with col2:
                if st.button("🔍 Key Findings"):
                    st.session_state.suggested_question = "What are the key findings or main conclusions presented in this document?"
            with col3:
                if st.button("📊 Main Topic"):
                    st.session_state.suggested_question = "What is the main topic or subject of this document?"
    
    # Handle suggested questions
    suggested_question = st.session_state.get('suggested_question', '')
    if suggested_question:
        user_question = st.text_input("Ask a question about your documents:", value=suggested_question, key="question_input")
        st.session_state.suggested_question = ''  # Clear after use
    else:
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
                                conversation_chain = get_conversation_chain(vectorstore, "HuggingFace FLAN-T5", "flan-t5-base")
                                
                                if conversation_chain is not None:
                                    st.session_state.conversation = conversation_chain
                                    st.success("✅ Documents processed successfully! You can now ask questions.")
                                else:
                                    st.error("Failed to create conversation chain. Please try again.")
                                
                    except Exception as e:
                        st.error(f"An error occurred while processing: {str(e)}")
                        st.error("Please try again with different PDF files.")

    # Model selection
    st.sidebar.header("Model Configuration")
    model_type = st.sidebar.radio(
        "Select Model Type:",
        ("OpenAI GPT", "HuggingFace FLAN-T5")
    )

    if model_type == "OpenAI GPT":
        model_version = st.sidebar.selectbox(
            "Select OpenAI Model Version:",
            ("gpt-4", "gpt-3.5-turbo")
        )
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    else:
        model_version = st.sidebar.selectbox(
            "Select HuggingFace Model Version:",
            ("flan-t5-base", "flan-t5-small")
        )

    # Tool selection
    tool_preference = st.sidebar.radio(
        "Select Tool Preference:",
        ("Local", "Online")
    )

    st.sidebar.write("\nConfiguration Summary:")
    st.sidebar.write(f"Model Type: {model_type}")
    st.sidebar.write(f"Model Version: {model_version}")
    st.sidebar.write(f"Tool Preference: {tool_preference}")

    # Set up the chatbot with the selected configuration
    chatbot = setup_chatbot(model_type, model_version, tool_preference)

    if chatbot:
        st.write("Chatbot is ready! Start asking questions.")
    else:
        st.error("Failed to set up the chatbot. Please check the configuration.")

def setup_chatbot(model_type, model_version, tool_preference):
    """Set up the chatbot based on user configuration."""
    st.write(f"Setting up chatbot with the following configuration:")
    st.write(f"Model Type: {model_type}")
    st.write(f"Model Version: {model_version}")
    st.write(f"Tool Preference: {tool_preference}")

    # Initialize vectorstore (mocked for now)
    vectorstore = st.session_state.get('vectorstore', None)
    if not vectorstore:
        st.error("Vectorstore is not initialized. Please process your documents first.")
        return None

    # Call get_conversation_chain with the selected model type and version
    return get_conversation_chain(vectorstore, model_type, model_version)

# Ensure proper handling of asyncio event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

if __name__ == '__main__':
    main()