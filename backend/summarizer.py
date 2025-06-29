from transformers import pipeline, AutoTokenizer
import os
import re

# Global variables for lazy initialization
summarizer = None
tokenizer = None
MAX_MODEL_TOKENS = 1024  # BART's max input tokens
CHUNK_TOKENS = 900  # Leave room for special tokens and summary

# Check for Hugging Face token
def check_token():
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token or token == "your_token_here":
        return False
    return True

# Initialize models only if token is available
def get_summarizer():
    global summarizer, tokenizer
    if not check_token():
        return None, None
    
    if summarizer is None or tokenizer is None:
        try:
            MODEL_NAME = "facebook/bart-large-cnn"
            summarizer = pipeline("summarization", model=MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except Exception as e:
            print(f"Error initializing summarizer: {e}")
            return None, None
    
    return summarizer, tokenizer

# Helper to split text into sentences
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_sentences(sentences, max_tokens=CHUNK_TOKENS):
    current_tokenizer = get_summarizer()[1]
    if not current_tokenizer:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = len(current_tokenizer.encode(sent, add_special_tokens=False))
        # If a single sentence is too long, truncate it
        if sent_tokens > max_tokens:
            sent = current_tokenizer.decode(current_tokenizer.encode(sent, add_special_tokens=False)[:max_tokens])
            sent_tokens = max_tokens
        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    # Remove empty or very short chunks
    return [c for c in chunks if len(c.strip()) > 20]

def generate_summary(text, max_length=200):
    if not check_token():
        return "Error: HuggingFace API token not configured. Please set HUGGINGFACEHUB_API_TOKEN in your Streamlit Cloud secrets."
    
    current_summarizer, current_tokenizer = get_summarizer()
    if not current_summarizer or not current_tokenizer:
        return "Error: Failed to initialize summarization model. Please check your API token and try again."
    
    if not text.strip():
        return "No text provided for summarization."
    
    try:
        sentences = split_into_sentences(text)
        chunks = chunk_sentences(sentences, max_tokens=CHUNK_TOKENS)
        summaries = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            # Truncate chunk if it exceeds model's max tokens
            input_ids = current_tokenizer.encode(chunk, add_special_tokens=False)
            if len(input_ids) > MAX_MODEL_TOKENS:
                chunk = current_tokenizer.decode(input_ids[:MAX_MODEL_TOKENS])
            try:
                summary = current_summarizer(chunk, max_length=max_length, truncation=True)[0]['summary_text']
            except Exception as e:
                summary = f"[Error summarizing chunk: {e}]"
            summaries.append(summary)
        # Optionally, summarize the summaries if there are many
        summaries = [s for s in summaries if s and not s.startswith('[Error')]
        if len(summaries) > 1:
            combined = " ".join(summaries)
            input_ids = current_tokenizer.encode(combined, add_special_tokens=False)
            if len(input_ids) > MAX_MODEL_TOKENS:
                combined = current_tokenizer.decode(input_ids[:MAX_MODEL_TOKENS])
            try:
                final_summary = current_summarizer(combined, max_length=max_length, truncation=True)[0]['summary_text']
            except Exception as e:
                final_summary = combined
            return final_summary
        return summaries[0] if summaries else "No summary generated."
    except Exception as e:
        return f"Error generating summary: {str(e)}"
