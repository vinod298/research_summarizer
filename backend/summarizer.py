from transformers import pipeline, AutoTokenizer
import os
import re

# Check for Hugging Face token
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is required. Please set it in your Streamlit Cloud secrets.")

MODEL_NAME = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_MODEL_TOKENS = 1024  # BART's max input tokens
CHUNK_TOKENS = 900  # Leave room for special tokens and summary

# Helper to split text into sentences
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_sentences(sentences, max_tokens=CHUNK_TOKENS):
    chunks = []
    current_chunk = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        # If a single sentence is too long, truncate it
        if sent_tokens > max_tokens:
            sent = tokenizer.decode(tokenizer.encode(sent, add_special_tokens=False)[:max_tokens])
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
    if not text.strip():
        return "No text provided for summarization."
    sentences = split_into_sentences(text)
    chunks = chunk_sentences(sentences, max_tokens=CHUNK_TOKENS)
    summaries = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        # Truncate chunk if it exceeds model's max tokens
        input_ids = tokenizer.encode(chunk, add_special_tokens=False)
        if len(input_ids) > MAX_MODEL_TOKENS:
            chunk = tokenizer.decode(input_ids[:MAX_MODEL_TOKENS])
        try:
            summary = summarizer(chunk, max_length=max_length, truncation=True)[0]['summary_text']
        except Exception as e:
            summary = f"[Error summarizing chunk: {e}]"
        summaries.append(summary)
    # Optionally, summarize the summaries if there are many
    summaries = [s for s in summaries if s and not s.startswith('[Error')]
    if len(summaries) > 1:
        combined = " ".join(summaries)
        input_ids = tokenizer.encode(combined, add_special_tokens=False)
        if len(input_ids) > MAX_MODEL_TOKENS:
            combined = tokenizer.decode(input_ids[:MAX_MODEL_TOKENS])
        try:
            final_summary = summarizer(combined, max_length=max_length, truncation=True)[0]['summary_text']
        except Exception as e:
            final_summary = combined
        return final_summary
    return summaries[0] if summaries else "No summary generated."
