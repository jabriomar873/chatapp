# Multiple PDF Chat Application 📚🤖

A powerful Streamlit-based application that allows you to chat with multiple PDF documents using AI. Supports both text-based and scanned PDFs with OCR capabilities.

## 🌟 Features

- **Multi-PDF Support**: Upload and process multiple PDF documents simultaneously
- **OCR Technology**: Handles scanned/image-based PDFs using Tesseract OCR
- **Mixed Content Support**: Processes PDFs with both text and scanned pages
- **Multi-language OCR**: Supports English, French, German, Spanish, Italian, and Portuguese
- **Local AI Model**: Uses local HuggingFace models (no API keys required)
- **Smart Text Extraction**: 3-tier extraction method (PyPDF → PyMuPDF → OCR)
- **Interactive Chat Interface**: Real-time conversation with your documents
- **Conversation Memory**: Maintains chat history throughout the session

## 🚀 Supported PDF Types

- ✅ **Text-based PDFs** - Regular PDFs with selectable text
- ✅ **Scanned PDFs** - Image-based PDFs (uses OCR)
- ✅ **Mixed PDFs** - PDFs with both text and scanned pages

## 🛠️ Installation

### Prerequisites
- Python 3.8 or later
- Windows/Linux/Mac

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gon-martinam/multiple-pdf-chat.git
   cd multiple-pdf-chat
   ```

2. **Install Python dependencies:**
   ```bash
   pip install streamlit pypdf langchain-huggingface langchain-community
   pip install faiss-cpu sentence-transformers transformers torch
   pip install PyMuPDF pillow pytesseract python-dotenv
   ```

3. **Install Tesseract OCR (for scanned PDFs):**
   
   **Windows:**
   ```bash
   winget install UB-Mannheim.TesseractOCR
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get install tesseract-ocr
   ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```

4. **Create environment file (optional):**
   ```bash
   touch .env
   ```

## 🎯 Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Configure OCR settings** (if needed):
   - Expand "🔍 OCR Settings" in the sidebar
   - Select your document language
   - Verify Tesseract installation status

4. **Upload PDFs:**
   - Use the file uploader in the sidebar
   - Select one or multiple PDF files
   - Click "Process" to analyze documents

5. **Start chatting:**
   - Ask questions about your documents
   - The AI will provide answers based on the content
   - Chat history is maintained throughout the session

## 💡 Example Questions

- "What is the main topic of this document?"
- "Summarize the key findings from all PDFs"
- "What are the conclusions in the research paper?"
- "Find information about [specific topic]"

## 🔧 Technical Architecture

- **Frontend**: Streamlit web interface
- **PDF Processing**: PyPDF, PyMuPDF, and Tesseract OCR
- **AI Models**: HuggingFace transformers (FLAN-T5, sentence-transformers)
- **Vector Database**: FAISS for document similarity search
- **Memory Management**: LangChain conversation buffer

## 🎨 Customization

The application uses custom HTML templates for the chat interface. You can modify `htmlTemplates.py` to customize the appearance of chat messages.

## 🔍 Troubleshooting

**OCR not working?**
- Ensure Tesseract is properly installed
- Restart the application after installing Tesseract
- Check the OCR settings panel for installation status

**Model loading issues?**
- Ensure stable internet connection for first-time model download
- Check that you have sufficient disk space for model files

**PDF processing errors?**
- Try different PDF files
- Ensure PDFs are not password-protected
- Check if PDFs contain readable content

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for the transformer models
- LangChain for the conversation framework
- Streamlit for the web interface
- Tesseract OCR for text recognition

## Contributing 🤝
If you want to contribute to the Multiple PDF Chatbot, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## License ⚖️
This project uses the following license: [MIT License](LICENSE).

If you have any questions or suggestions, feel free to open an issue or submit a pull request. Happy chatting! 😃
