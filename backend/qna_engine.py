from transformers import pipeline
import os

# Check for Hugging Face token
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is required. Please set it in your Streamlit Cloud secrets.")

# Initialize the text2text generation model (FLAN-T5)
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def ask_question(context, question):
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
