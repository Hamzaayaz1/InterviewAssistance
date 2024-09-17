import dspy
import os
from config import setup_cohere_client, setup_logging, setup_llama_parser
from utils import parse_pdf, truncate_text
from dspy.teleprompt import BootstrapFewShot
import logging
# from question_generation import InterviewQuestionGenerator

# Initial setup functions
setup_llama_parser()
setup_cohere_client()
setup_logging()


print(setup_cohere_client())
logger = logging.getLogger(__name__)

# File paths
resume_file = "D:\\MyProjects\\InterviewAssistance\\ResumeCV1.pdf"
job_file = "D:\\MyProjects\\InterviewAssistance\\jobdiscr.pdf"

# resume_text = truncate_text(resume_parse)
# job_text = truncate_text(job_parse)

# Create training examples
def create_train_example(resume_text, job_text, last_answer, previous_questions, question):
    return dspy.Example(
        resume_text=resume_text,
        job_text=job_text,
        previous_answers=last_answer,
        previous_questions=previous_questions,
        question=question
    )
resume_text = parse_pdf(resume_file, "resume")
job_text = parse_pdf(job_file, "job_disc")

train_examples = [
    create_train_example(resume_text, job_text, "", [],
                         "Welcome to the Interview Phil. From the job text, this Senior Java developer role requires Spring Boot for Microservices, J-unit for testing, and SQL for database management. Do you have these skills?"),
    create_train_example(resume_text, job_text,
                         "Yes, I have all three skills. I gained Spring Boot experience in HRMS at TrueLancer, J-Unit at HTC-Global, and SQL at Wipro.",
                         "Welcome to the Interview Phil. From the job text, this Senior Java developer role requires Spring Boot for Microservices, J-unit for testing, and SQL for database management. Do you have these skills?",
                         "Can you describe how you used Spring Boot for scalability in Microservices?"),
    create_train_example(resume_text, job_text,
                         "At TrueLancer, I designed Spring Boot microservices, improving system scalability.",
                         ["Welcome to the Interview Phil. From the job text, this Senior Java developer role requires Spring Boot for Microservices, J-unit for testing, and SQL for database management. Do you have these skills?",
                          "Can you describe how you used Spring Boot for scalability in Microservices?"],
                         "Tell me about a project where your modular coding made a difference."),
    create_train_example(resume_text, job_text,
                         "I refactored TrueLancer's monolithic codebase into microservices, reducing deployment cycles by 3 days.",
                         ["Welcome to the Interview Phil. From the job text, this Senior Java developer role requires Spring Boot for Microservices, J-unit for testing, and SQL for database management. Do you have these skills?",
                          "Can you describe how you used Spring Boot for scalability in Microservices?",
                          "Tell me about a project where your modular coding made a difference."],
                         "How did Spring Boot's auto-configuration feature support faster deployments?")
]

print("Training examples created.")

# Trainset preparation
trainset = [ex.with_inputs('resume_text', 'job_text', 'previous_answers', 'previous_questions') for ex in train_examples]

print("TrainSet Created.")

# Define Assessment class
class Assess(dspy.Signature):
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

# Skill Keywords
skill_keywords = {
    "sql": ["sql"],
    "python": ["python"],
    "powerbi": ["powerbi"],
}

def identify_current_skill(question, previous_questions):
    question_text = question.lower()
    for skill, keywords in skill_keywords.items():
        if any(keyword in question_text or any(keyword in q.lower() for q in previous_questions) for keyword in keywords):
            return skill
    return None

# Metric function
def metric(gold, pred, trace=None):
    current_skill = identify_current_skill(pred.question, gold.previous_questions)

    # Using Cohere client in context
    with dspy.context(lm=setup_cohere_client()):
        # Check for redundancy
        question_count_check = f"Have there already been 3 questions asked about {current_skill}?"
        question_count_check_result = dspy.Predict(Assess)(
            assessed_text="\n".join(gold.previous_questions),
            assessment_question=question_count_check
        )
        is_redundant = question_count_check_result.assessment_answer.lower() == "yes"

        # Check for relevance to job description
        relevance_check = "Is this question relevant to the job description?"
        relevance_check_result = dspy.Predict(Assess)(
            assessed_text=pred.question,
            assessment_question=relevance_check
        )
        is_relevant = relevance_check_result.assessment_answer.lower() == "yes"

        # Check for appropriate difficulty
        difficulty_check = "Is this question appropriately challenging for a senior developer role?"
        difficulty_check_result = dspy.Predict(Assess)(
            assessed_text=pred.question,
            assessment_question=difficulty_check
        )
        is_appropriate_difficulty = difficulty_check_result.assessment_answer.lower() == "yes"

    # Calculate score based on multiple factors
    score = 0
    if not is_redundant:
        score += 1
    if is_relevant:
        score += 1
    if is_appropriate_difficulty:
        score += 1

    # Normalize score to be between 0 and 1
    normalized_score = score / 3

    if trace is not None:
        return normalized_score > 0.5
    return normalized_score

print("Metric created.")

# Model evaluation
def evaluate_model(module):
    evaluator = dspy.evaluate.Evaluate(
        devset=trainset,
        num_threads=1,
        display_progress=True,
        display_table=5
    )
    print("\nEvaluator object created.\n")
    evaluation_score = evaluator(module, metric)
    print("\nEvaluation completed.\n")
    print(f"\nAverage Metric:\n{evaluation_score}\n")
    return evaluation_score

# Bootstrapping and saving the compiled module
def compile_and_save_module(compile_module, resume_text, job_text, 
                            previous_questions = [], 
                            previous_answers = []):
    logger.debug(f"compile_and_save_module called with: previous_questions={previous_questions}, type={type(previous_questions)}")
    compiled_module_path = "D:\\MyProjects\\InterviewAssistance\\compiled_interview_module.json"
    
    config = dict(max_bootstrapped_demos=5, max_labeled_demos=5)

    teleprompter = BootstrapFewShot(metric=metric, **config)

    
    if os.path.exists(compiled_module_path) and os.path.getsize(compiled_module_path) > 0:

        compile_module.load(compiled_module_path)
    else:
        compile_module = teleprompter.compile(student=compile_module, trainset=trainset)
        compile_module.save(compiled_module_path)

    return compile_module(resume_text, job_text, previous_questions, previous_answers)

print("Training Completed!")
