import streamlit as st
from backend.summarizer import generate_summary
from backend.qna_engine import ask_question, generate_logic_questions
import os
import PyPDF2
import io

# Page configuration
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Check for HuggingFace token
@st.cache_data
def check_hf_token():
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token or token == "your_token_here":
        st.error("⚠️ HuggingFace API token not found! Please set HUGGINGFACEHUB_API_TOKEN in your environment variables.")
        st.info("Get your token from: https://huggingface.co/settings/tokens")
        return False
    return True

# Main app
def main():
    st.markdown('<h1 class="main-header">📄 Smart Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Upload a document, get a summary, and ask questions!")
    
    # Check API token
    if not check_hf_token():
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your research document (.txt or .pdf)", 
        type=["txt", "pdf"],
        help="Supported formats: PDF and TXT files up to 200MB"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        st.json(file_details)
        
        # Process file
        try:
            if uploaded_file.name.endswith(".pdf"):
                # Reset file pointer
                uploaded_file.seek(0)
                reader = PyPDF2.PdfReader(uploaded_file)
                doc_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                if not doc_text.strip():
                    st.error("Could not extract text from PDF. Please ensure the PDF contains readable text.")
                    return
            else:
                doc_text = uploaded_file.read().decode("utf-8")
            
            # Display document preview
            with st.expander("📖 Document Preview (First 500 characters)"):
                st.text(doc_text[:500] + "..." if len(doc_text) > 500 else doc_text)
            
            # Summary section
            st.markdown('<h2 class="sub-header">📌 Document Summary</h2>', unsafe_allow_html=True)
            if st.button("Generate Summary", type="primary"):
                with st.spinner("🤖 Generating summary..."):
                    try:
                        summary = generate_summary(doc_text)
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ Summary generated successfully!")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.write(summary)
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
            
            # Q&A section
            st.markdown('<h2 class="sub-header">❓ Ask a Question</h2>', unsafe_allow_html=True)
            query = st.text_input("Enter your question about the document:")
            if st.button("Get Answer") and query:
                with st.spinner("🔍 Searching for answer..."):
                    try:
                        response = ask_question(doc_text, query)
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ Answer found!")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.write(response)
                    except Exception as e:
                        st.error(f"❌ Error getting answer: {str(e)}")
            
            # Logic questions section
            st.markdown('<h2 class="sub-header">🧠 Logic-Based Questions</h2>', unsafe_allow_html=True)
            if st.button("Generate Logic-Based Questions"):
                with st.spinner("🧠 Creating comprehension questions..."):
                    try:
                        questions = generate_logic_questions(doc_text)
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ Questions generated successfully!")
                        st.markdown('</div>', unsafe_allow_html=True)
                        for i, q in enumerate(questions, 1):
                            st.markdown(f"**{i}.** {q}")
                    except Exception as e:
                        st.error(f"❌ Error generating questions: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Sidebar with instructions
    with st.sidebar:
        st.markdown("## 📋 Instructions")
        st.markdown("""
        1. **Upload** a PDF or TXT file
        2. **Generate Summary** for a concise overview
        3. **Ask Questions** about specific content
        4. **Generate Logic Questions** for comprehension practice
        """)
        
        st.markdown("## ⚙️ Settings")
        st.info("Make sure to set your HuggingFace API token in the environment variables.")
        
        st.markdown("## 🔗 Links")
        st.markdown("[Get HuggingFace Token](https://huggingface.co/settings/tokens)")
        st.markdown("[View Source Code](https://github.com/yourusername/research_summarizer)")

if __name__ == "__main__":
    main()
