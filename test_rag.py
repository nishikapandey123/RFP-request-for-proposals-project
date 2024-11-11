import os
import openai
from query_data import query_rag
from extract_questions import main as extract_questions_main

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def query_and_validate(question: str, expected_response: str, api_key: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    openai.api_key = api_key
    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=60
    )
    evaluation_results_str_cleaned = response.choices[0].text.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

def extract_and_validate_rfp(file_path: str, api_key: str):
    questions = extract_questions_main(file_path)
    for question in questions:
        expected_response = "expected response placeholder"
        print(f"Question: {question}")
        print(f"Validation Result: {query_and_validate(question, expected_response, api_key)}")

def get_rfp_file_paths(directory: str):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pdf')]

if __name__ == "__main__":
    rfp_directory = "C://Users//Lenovo//ragcreate//rag-tutorial-v2//data"
    api_key = "sk-proj-4xs34D7dbhQs1STxe7sHT3BlbkFJr3uy6Exdr6UwlpfhAHAN"

    rfp_file_paths = get_rfp_file_paths(rfp_directory)

    for file_path in rfp_file_paths:
        extract_and_validate_rfp(file_path, api_key)
