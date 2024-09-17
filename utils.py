import os
import tempfile
import streamlit as st
from llama_parse import LlamaParse
from config import setup_llama_parser, setup_logging
import logging

setup_logging()

def truncate_text(text, max_tokens=2048):
    """Truncate text to a maximum number of tokens (words)."""
    words = text.split()
    return ' '.join(words[:max_tokens])

@st.cache_data(ttl=1200)
def parse_pdf(uploaded_file, file_type):
    """Parse a PDF file using LlamaParse."""
    if uploaded_file is None:
        return None

    try:
        parser = setup_llama_parser()

        if isinstance(uploaded_file, str):
            # If uploaded_file is a string, assume it's a file path
            temp_path = uploaded_file
        else:
            # If uploaded_file is a file-like object (e.g., from st.file_uploader)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

        # with st.spinner(f"Parsing {file_type.capitalize()}..."):
        documents = parser.load_data(temp_path)
        if not documents:
            raise ValueError("No documents were parsed")
        parsed_text = documents[0].text

        # Clean up the temporary file only if we created one
        if not isinstance(uploaded_file, str):
            os.unlink(temp_path)

        return parsed_text
    except Exception as e:
        logging.error(f"Error parsing {file_type} with LlamaParse: {str(e)}")
        return None

def safe_parse_pdf(uploaded_file, file_type):
    """Safely parse a PDF file, returning an empty string if parsing fails."""
    parsed_text = parse_pdf(uploaded_file, file_type)
    return parsed_text if parsed_text is not None else ""