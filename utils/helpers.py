def extract_paragraph_references(text, answer):
    # Naive matching example
    paras = text.split("\n\n")
    for i, para in enumerate(paras):
        if answer.strip()[:20] in para:
            return f"Paragraph {i + 1}"
    return "Source paragraph not clearly found."
