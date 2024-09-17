import logging
from dotenv import load_dotenv
import os
import dspy
from dspy import Cohere
from cohere.errors import UnauthorizedError
from llama_parse import LlamaParse

load_dotenv()


def setup_logging():
    logging.basicConfig(level=logging.INFO)


def setup_llama_parser():
    llama_parse_key = os.environ.get("llama_parse_key")
    if not llama_parse_key:
        logging.error("LlamaParse API key not found in environment variables")
        raise ValueError("LlamaParse API key is missing. Please check your .env file.")
    logging.info(f"Successfully loaded LlamaParse API key. First 5 characters: {llama_parse_key[:5]}...")
    try:
        return LlamaParse(api_key=llama_parse_key, result_type="text", verbose=True)
    except Exception as e:
        logging.error(f"Error initializing LlamaParse: {str(e)}")
        raise


def setup_cohere_client():
    # Initialize cohere client api
    cohere_api_key = os.environ.get("cohere_api_key")
    if not cohere_api_key:
        logging.error("Cohere API key not found in environment variables")
        raise ValueError("Cohere API key is missing. Please check your .env file.")

    logging.info(f"Cohere API key loaded. First 5 characters: {cohere_api_key[:5]}...")

    try:
        coh = Cohere(model='command-r', api_key=cohere_api_key)
        dspy.settings.configure(lm=coh)
        logging.info("Cohere client initialized successfully")
        return coh

    except UnauthorizedError as e:
        logging.error(f"Unauthorized error when initializing Cohere: {str(e)}")
        raise ValueError("Invalid Cohere API key. Please check your API key and try again.") from e

    except Exception as e:
        logging.error(f"Error initializing Cohere: {str(e)}")
        raise
        # print(f"Error initializing Cohere: {e}")

def test_coh(coh):
    # Test the Cohere client
    try:
        test_response = coh("This is a test query to check if the Cohere client is working.")
        logging.info("Cohere client test successful")
    except Exception as e:
        logging.error(f"Error testing Cohere client: {str(e)}")
        raise
