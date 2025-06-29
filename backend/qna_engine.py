from transformers import pipeline
import os

# Check for Hugging Face token
def check_token():
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token or token == "your_token_here":
        return False
    return True

# Initialize the text2text generation model (FLAN-T5)
def get_llm_pipeline():
    if not check_token():
        return None
    
    try:
        llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
        return llm_pipeline
    except Exception as e:
        print(f"Error initializing Q&A pipeline: {e}")
        return None

llm_pipeline = get_llm_pipeline()

def ask_question(context, question):
    if not check_token():
        return "Error: HuggingFace API token not configured. Please set HUGGINGFACEHUB_API_TOKEN in your Streamlit Cloud secrets."
    
    if not llm_pipeline:
        return "Error: Failed to initialize Q&A model. Please check your API token and try again."
    
    # Truncate context if too long
    max_context_length = 1000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    try:
        result = llm_pipeline(prompt, max_length=256, do_sample=True, temperature=0.7)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def generate_logic_questions(text, count=5):
    if not check_token():
        return ["Error: HuggingFace API token not configured. Please set HUGGINGFACEHUB_API_TOKEN in your Streamlit Cloud secrets."]
    
    if not llm_pipeline:
        return ["Error: Failed to initialize Q&A model. Please check your API token and try again."]
    
    # Truncate text if too long
    max_text_length = 800
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."
    
    prompt = f"Generate {count} comprehension questions from the following text:\n\n{text}\n\nQuestions:"
    try:
        results = llm_pipeline(prompt, max_length=512, do_sample=True, temperature=0.8)
        questions = results[0]['generated_text'].split("\n")
        # Clean up questions
        cleaned_questions = []
        for q in questions:
            q = q.strip("-•1234567890. ")
            if q and len(q) > 10:  # Only keep meaningful questions
                cleaned_questions.append(q)
        return cleaned_questions[:count]  # Return only requested number
    except Exception as e:
        return [f"Error generating questions: {str(e)}"]
