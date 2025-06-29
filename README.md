# 📄 Smart Research Assistant

An AI-powered assistant for research documents. Upload a PDF or TXT file, get a concise summary, ask questions, and generate logic-based questions—all in your browser.

---

## 🚀 Features
- **Document Upload**: Supports PDF and TXT files.
- **Automatic Summarization**: Uses state-of-the-art models to summarize long documents.
- **Ask Anything**: Get answers to your questions based on the uploaded document.
- **Logic-Based Questions**: Generate logic/comprehension questions from the document.
- **Answer Evaluation**: (Advanced) Evaluate answers with justification and paragraph reference.

---

## 🧩 Architecture
- **Frontend**: Streamlit (app.py)
- **Backend**: Python modules in `/backend`:
  - `summarizer.py`: Summarizes documents using HuggingFace Transformers (BART model).
  - `qna_engine.py`: Q&A and logic question generation (FLAN-T5 model).
  - `processor.py`: Extracts text from PDF/TXT files.
  - `evaluator.py`: (Optional) Evaluates answers using OpenAI GPT-3.5-turbo (via LangChain).
- **Utils**: Helper functions in `/utils`.
- **Data**: Uploaded files are stored in `/data/uploads`.

---

## 🛠 Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies:
  - streamlit
  - transformers
  - huggingface_hub
  - torch
  - PyPDF2

**Optional for PDF extraction:**
- `PyMuPDF` (fitz) for advanced PDF processing (used in backend/processor.py)
- `langchain`, `openai`, `python-dotenv` for answer evaluation (see backend/evaluator.py)

---

## 🔑 API Keys
- **HuggingFace**: Required for summarization and Q&A. Set the environment variable `HUGGINGFACEHUB_API_TOKEN` with your HuggingFace access token.
- **OpenAI**: (Optional, for evaluation) Set `OPENAI_API_KEY` in your environment or `.env` file.

---

## ▶️ Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/genai_doc_assistant.git
   cd genai_doc_assistant
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your HuggingFace token**
   - Option 1: In your shell before running the app:
     ```bash
     export HUGGINGFACEHUB_API_TOKEN=your_token_here
     ```
   - Option 2: Edit the placeholder in `backend/summarizer.py` and `backend/qna_engine.py`.

4. **(Optional) Set your OpenAI key**
   - Create a `.env` file with:
     ```
     OPENAI_API_KEY=your_openai_key_here
     ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📝 Usage
- Upload a `.pdf` or `.txt` file.
- Click **Generate Summary** to get a concise summary.
- Ask questions in the **Ask a Question** box.
- Click **Generate Logic-Based Questions** for practice/comprehension questions.
- (If enabled) Use the evaluation feature for answer feedback.

---

## 📂 Project Structure
```
research_summarizer/
├── app.py                # Streamlit frontend
├── backend/
│   ├── summarizer.py     # Summarization logic
│   ├── qna_engine.py     # Q&A and logic question generation
│   ├── processor.py      # Document text extraction
│   └── evaluator.py      # (Optional) Answer evaluation
├── utils/
│   └── helpers.py        # Helper functions
├── data/
│   └── uploads/          # Uploaded files
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🤝 Contributing
Pull requests and suggestions are welcome!

---

## 📃 License
MIT License
