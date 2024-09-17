import dspy
from dspy import Cohere
from dspy.teleprompt import BootstrapFewShot
import streamlit as st
from cohere.errors import TooManyRequestsError, BadRequestError, UnauthorizedError
import time
import logging
from config import setup_cohere_client, setup_llama_parser, setup_logging
from utils import truncate_text
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from dspy import InputField, OutputField


setup_cohere_client()
setup_logging()
setup_llama_parser()
print(setup_cohere_client())
logger = logging.getLogger(__name__)


class GenerateInterviewQuestion(dspy.Signature):
    """Generate tailored interview questions based on resume, job description, and previous answers."""
    # resume_text = dspy.InputField(desc="Detailed text of the candidate's resume")
    # job_text = dspy.InputField(desc="Comprehensive text of the job description")
    # previous_questions = dspy.InputField(desc="String of previously asked interview questions, separated by newlines")
    # previous_answers = dspy.InputField(desc="String of candidate's previous answers, separated by newlines")
    # question = dspy.OutputField(desc="Thoughtful interview question")
    # rationale = dspy.OutputField(desc="Brief explanation of the question's relevance")

    resume_text: str = InputField(description="Detailed text of the candidate's resume")
    job_text: str = InputField(description="Comprehensive text of the job description")
    previous_questions: str = InputField(default="", description="String of previously asked interview questions, separated by newlines")
    previous_answers: str = InputField(default="", description="String of candidate's previous answers, separated by newlines")
    
    question: str = OutputField(description="Thoughtful interview question")
    rationale: str = OutputField(description="Brief explanation of the question's relevance")


class InterviewQuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_question = dspy.Predict(GenerateInterviewQuestion)
    
    def forward(self, resume_text, job_text, previous_questions=[], previous_answers=[]):
        logger.debug(f"InterviewQuestionGenerator.forward called with: previous_questions={previous_questions}, type={type(previous_questions)}")
        # Convert lists to strings
        previous_questions_str = "\n".join(previous_questions) if previous_questions else ""
        previous_answers_str = "\n".join(previous_answers) if previous_answers else ""
        
        prediction = self.generate_question(
            resume_text=resume_text, 
            job_text=job_text, 
            previous_questions=previous_questions_str,
            previous_answers=previous_answers_str
        )
        
        return dspy.Prediction(question=prediction.question, rationale=prediction.rationale)


