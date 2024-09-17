import streamlit as st
from question_generation import InterviewQuestionGenerator

from cohere.errors import TooManyRequestsError, BadRequestError, UnauthorizedError
import time
from config import setup_logging
import logging
from compile_module import compile_and_save_module, evaluate_model

setup_logging()

# @st.cache_data(ttl=3600, max_entries=9)
def rate_limited_generate_question(
    resume_text,
    job_text,
    previous_questions = [],
    previous_answers=[]
) :
    # previous_questions_str = " ".join(previous_questions) if previous_questions else ""
    # previous_answers_str = " ".join(previous_answers) if previous_answers else ""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # if generator is None:
            generator = InterviewQuestionGenerator()
            result = compile_and_save_module(
                compile_module=generator,
                resume_text=resume_text,
                job_text=job_text,
                previous_questions=previous_questions,
                previous_answers=previous_answers
            )
            evaluator = evaluate_model(generator)
            print(evaluator)
            return result.question, result.rationale
        except UnauthorizedError as e:
            logging.error(f"Unauthorized error: {str(e)}")
            # st.error("Invalid API key. Please check your Cohere API key and try again.")
            raise
        except TooManyRequestsError:
            if attempt < max_retries - 1:
                wait_time = 70 * (attempt + 1)
                # st.warning(f"Rate limit reached. Waiting for {wait_time} seconds before trying again.")
                logging.warning(f"Rate limit reached. Waiting for {wait_time} seconds before trying again.")
                time.sleep(wait_time)
            else:
                # st.error("Rate limit exceeded. Please try again later.")
                logging.error("Rate limit exceeded. Please try again later.")
                raise
        except BadRequestError as e:
            # st.error(f"Error generating question: {str(e)}")
            logging.error(f"Error generating question: {str(e)}")
            return None, None
        except Exception as e:
            logging.error(f"Unexpected error in rate_limited_generate_question: {str(e)}")
            # st.error(f"An unexpected error occurred: {str(e)}")
            raise

    return None, None  # If all retries fa