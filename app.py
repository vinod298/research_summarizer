import streamlit as st
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
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
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
        return False
    return True

# Import backend modules only if token is available
def import_backend_modules():
    try:
        if check_hf_token():
            from backend.summarizer import generate_summary
            from backend.qna_engine import ask_question, generate_logic_questions
            return generate_summary, ask_question, generate_logic_questions
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error importing backend modules: {str(e)}")
        return None, None, None

# Main app
def main():
    st.markdown('<h1 class="main-header">📄 Smart Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Upload a document, get a summary, and ask questions!")
    
    # Check API token and import modules
    if not check_hf_token():
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("⚠️ HuggingFace API token not found!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("""
        **To fix this issue:**
        
        1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens) to get your API token
        2. In your Streamlit Cloud app settings, go to **Secrets**
        3. Add the following:
        ```
        HUGGINGFACEHUB_API_TOKEN = "hf_your_actual_token_here"
        ```
        4. Save and redeploy your app
        """)
        
        st.markdown("### 🔗 Quick Links")
        st.markdown("- [Get HuggingFace Token](https://huggingface.co/settings/tokens)")
        st.markdown("- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)")
        
        return
    
    # Import backend modules
    generate_summary, ask_question, generate_logic_questions = import_backend_modules()
    
    if not all([generate_summary, ask_question, generate_logic_questions]):
        st.error("❌ Failed to load AI models. Please check your API token and try again.")
        return
    
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
        st.success("✅ HuggingFace API token is configured!")
        
        st.markdown("## 🔗 Links")
        st.markdown("[Get HuggingFace Token](https://huggingface.co/settings/tokens)")
        st.markdown("[View Source Code](https://github.com/yourusername/research_summarizer)")

if __name__ == "__main__":
    main()
