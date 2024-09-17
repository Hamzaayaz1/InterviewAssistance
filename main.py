import streamlit as st
from rate_limit_generate_question import rate_limited_generate_question
import logging
from utils import truncate_text, safe_parse_pdf

st.set_page_config(page_title="AI Interview Assistant", layout="wide")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'interview_completed' not in st.session_state:
        st.session_state.interview_completed = False
    if 'waiting_for_answer' not in st.session_state:
        st.session_state.waiting_for_answer = False



def start_interview(resume, job_desc):
    with st.spinner(f"Parsing ..."):
        resume = safe_parse_pdf(resume, "resume")
        job_desc = safe_parse_pdf(job_desc, "job_desc")
        st.session_state.parsed_resume = truncate_text(resume)
        st.session_state.parsed_job_desc = truncate_text(job_desc)
    
    if st.session_state.parsed_resume and st.session_state.parsed_job_desc:
        st.session_state.interview_started = True
        st.session_state.chat_history.append(("assistant", "Hello! I'm your AI interviewer today. Let's begin with the first question."))
        generate_next_question()
    else:
        st.error("Failed to parse uploaded files. Please try again.")

@st.cache_data(ttl=3600, max_entries=9)
def generate_next_question():
    previous_questions = [q.split("Q: ", 1)[1] for q, _ in st.session_state.chat_history if q.startswith("Q:")]
    previous_answers = [a for _, a in st.session_state.chat_history if not a.startswith("Q:")]
    
    # Join previous_questions into a single string
    previous_questions_str = "\n".join(previous_questions) if previous_questions else ""
    # Join previous_answers into a single string
    previous_answers_str = "\n".join(previous_answers) if previous_answers else ""

    logger.debug(f"Previous questions (string): {previous_questions_str}")
    logger.debug(f"Previous answers (string): {previous_answers_str}")

    try:
        question, rationale = rate_limited_generate_question(
            st.session_state.parsed_resume,
            st.session_state.parsed_job_desc,
            previous_questions,
            previous_answers
        )
        logger.debug(f"Generated question: {question}")
        logger.debug(f"Generated rationale: {rationale}")
    except Exception as e:
        logger.error(f"Error in rate_limited_generate_question: {str(e)}")
        raise

    if question:
        st.session_state.current_question = f"Q: {question}"
        st.session_state.chat_history.append(("assistant", st.session_state.current_question))
        st.session_state.waiting_for_answer = True
    else:
        st.session_state.interview_completed = True
        st.session_state.chat_history.append(("assistant", "Thank you for completing the interview. Let me know if you have any questions!"))

def display_chat_interface():
    st.write("### Interview Chat")
    
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    if not st.session_state.interview_completed:
        if st.session_state.waiting_for_answer:
            user_response = st.chat_input("Your answer")
            if user_response:
                st.session_state.chat_history.append(("human", user_response))
                st.session_state.waiting_for_answer = False
                st.rerun()
        else:
            with st.spinner("Generating next question..."):
                generate_next_question()
            st.rerun()
    else:
        if st.button("Start New Interview"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def main():
    
    st.title("AI Interview Assistant")

    initialize_session_state()

    with st.sidebar:
        uploaded_resume = st.file_uploader("Upload your resume (PDF only):", type="pdf")
        uploaded_job_desc = st.file_uploader("Upload the job description (PDF only):", type="pdf")
        
        if uploaded_resume and uploaded_job_desc:
            if st.button("Start Interview"):
                start_interview(uploaded_resume, uploaded_job_desc)

    if st.session_state.interview_started:
        display_chat_interface()

if __name__ == "__main__":
    main()