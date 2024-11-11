from flask import Flask, request, jsonify
from extract_questions import generate_questions_from_text, generate_content_using_prompt, extract_text_from_eligibility_section_from_link, extract_full_text_from_link
from populate_database import calculate_chunk_ids, split_documents, add_to_chroma
import os
import openai
import logging
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from langchain.schema import Document
from pymongo import MongoClient
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor  # Import for parallel processing

app = Flask(__name__)
app.secret_key = os.urandom(24)

openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

CHROMA_PATH = "chroma"

# MongoDB connection
client = MongoClient('mongodb://admin:DTU-Mongo@13.200.248.222:27017/')
db = client['generation']  # Replace with your actual database name
collection = db['content']  # Replace with your actual collection name

sections = {
    'okr': {'prompt': "Based on the following RFP content, write detailed Objectives and Key Results (OKRs) that align with the client's goals and expectations. Ensure each objective is clear, measurable, and includes key results that define success. Provide a rationale for each OKR to explain how it supports the overall project strategy.\n\nRFP Content:\n{rfp_content}", 'title': 'Objectives and Key Results (OKRs)'},
    'assumptions_constraints': {'prompt': "List the assumptions and constraints for the following RFP content. Ensure that each assumption is clearly stated, and describe any potential impact on the project scope, timeline, or budget. For each constraint, provide a detailed explanation of how it may affect the project and any mitigation strategies.\n\nRFP Content:\n{rfp_content}", 'title': 'Assumptions and Constraints'},
    'scope': {'prompt': "Define the scope of the project, specifying what is included and excluded, based on the following RFP content. Make sure to cover all critical aspects, including deliverables, timelines, and any specific constraints mentioned in the RFP. Provide detailed explanations for each element of the scope to ensure clarity and alignment with the client's expectations.\n\nRFP Content:\n{rfp_content}", 'title': 'Scope'},
    'architecture': {'prompt': "Describe the proposed technical architecture for the solution, considering the client's current infrastructure, future scalability, and security requirements, based on the following RFP content. Include diagrams or visual representations where appropriate, and explain how each component will interact to achieve the desired outcomes.\n\nRFP Content:\n{rfp_content}", 'title': 'Architecture'},
    'about_us': {'prompt': "Provide a brief overview of the company's background, expertise, and relevant experience in the IT services domain, based on the following RFP content. Highlight any unique strengths, certifications, or industry recognition that make the company a strong candidate for this project.\n\nRFP Content:\n{rfp_content}", 'title': 'About Us'},
    'out_of_scope': {'prompt': "Explicitly state what is NOT included in the project scope based on the following RFP content. Make sure to detail any exclusions that are critical for the client's understanding and avoid any potential misunderstandings.\n\nRFP Content:\n{rfp_content}", 'title': 'Out of Scope'},
    'deliverables': {'prompt': "List the tangible deliverables that will be produced as a result of the project, based on the following RFP content. For each deliverable, provide a description of its purpose, format, and the criteria for acceptance by the client.\n\nRFP Content:\n{rfp_content}", 'title': 'Deliverables'},
    'solution': {'prompt': "Provide a comprehensive description of the proposed solution, detailing how it addresses the objectives and requirements outlined in the following RFP content. Break down the solution into its core components, explaining the rationale behind each and how they work together to meet the client's needs. Include any innovative approaches or technologies that differentiate this solution.\n\nRFP Content:\n{rfp_content}", 'title': 'Solution'},
    'delivery_plan': {'prompt': "Outline a detailed project timeline, including key milestones, deliverables, and deadlines, based on the following RFP content. Explain how each phase of the project will be managed to ensure timely delivery, and address any potential bottlenecks or challenges. Include contingency plans for any critical paths in the timeline.\n\nRFP Content:\n{rfp_content}", 'title': 'Delivery Plan'},
    'tech_stack': {'prompt': "List the technologies, tools, and platforms that will be used in the project, based on the following RFP content. For each technology, explain why it was chosen and how it will contribute to the success of the project.\n\nRFP Content:\n{rfp_content}", 'title': 'Tech Stack'},
    'payment_terms': {'prompt': "Specify the payment terms and conditions for the project, including payment schedule, invoicing details, and late payment fees, based on the following RFP content. Ensure that all financial terms are clear, fair, and aligned with the client's expectations.\n\nRFP Content:\n{rfp_content}", 'title': 'Payment Terms'},
    'costing': {'prompt': "Provide a detailed breakdown of the project costs, including labor, materials, and any other relevant expenses, based on the following RFP content. Explain the rationale behind the costing structure and how it ensures value for the client while maintaining project profitability.\n\nRFP Content:\n{rfp_content}", 'title': 'Costing'},
    'escalation_matrix': {'prompt': "Define the escalation process for addressing any issues or concerns that arise during the project, based on the following RFP content. Specify the levels of escalation, the roles responsible at each level, and the expected response times.\n\nRFP Content:\n{rfp_content}", 'title': 'Escalation Matrix (L-1 of ESC)'},
    'key_contacts': {'prompt': "List the primary contacts for both the client and the project team, including names, roles, and contact information, based on the following RFP content. For each contact, provide a brief description of their role and responsibilities within the project.\n\nRFP Content:\n{rfp_content}", 'title': 'Key Contacts'},
    'slas': {'prompt': "Define the service level agreements (SLAs) for the project, specifying the expected levels of performance, availability, and support, based on the following RFP content. For each SLA, provide metrics and thresholds that will be used to measure compliance, and explain how these SLAs align with the client's expectations and industry standards.\n\nRFP Content:\n{rfp_content}", 'title': 'SLAs'},
    'risks_mitigation': {'prompt': "Identify potential risks that could impact the project, considering factors such as technical challenges, resource availability, and client dependencies, based on the following RFP content. For each risk, outline a mitigation plan that includes specific actions to reduce or eliminate the risk, and explain the rationale behind each mitigation strategy.\n\nRFP Content:\n{rfp_content}", 'title': 'Risks and Mitigation'},
    'wireframing': {'prompt': "Create visual representations of the user interface for the proposed solution based on the following RFP content. Provide detailed explanations for each wireframe, including the design rationale and how it supports the overall user experience.\n\nRFP Content:\n{rfp_content}", 'title': 'Wireframing'},
    'support_maintenance': {'prompt': "Outline the post-implementation support and maintenance services that will be provided, based on the following RFP content. Detail the scope of support, the response times, and the process for handling issues or updates.\n\nRFP Content:\n{rfp_content}", 'title': 'Support and Maintenance Plan'},
    'case_studies': {'prompt': "Showcase the company's past successes in delivering similar projects, based on the following RFP content. For each case study, provide a summary of the project, the challenges faced, the solutions provided, and the results achieved.\n\nRFP Content:\n{rfp_content}", 'title': 'Case Studies'},
    'references': {'prompt': "Provide contact information for references who can attest to the company's capabilities, based on the following RFP content. For each reference, include the project they were involved in, their role, and their contact details.\n\nRFP Content:\n{rfp_content}", 'title': 'References'},
    'annexures': {'prompt': "Include any additional supporting documents, such as technical specifications, resumes, or legal agreements, based on the following RFP content. For each document, explain its relevance and importance to the overall proposal.\n\nRFP Content:\n{rfp_content}", 'title': 'Annexures'},
    'scope_change': {'prompt': "Define the process for requesting and approving changes to the project scope or functional requirements, based on the following RFP content. Ensure that the process is clear, structured, and includes steps for documenting, evaluating, and communicating changes.\n\nRFP Content:\n{rfp_content}", 'title': 'Scope Change/FR Process'},
    'cr_process': {'prompt': "Define the process for addressing and resolving customer complaints or issues, based on the following RFP content. Include steps for logging complaints, investigating issues, and providing timely resolutions.\n\nRFP Content:\n{rfp_content}", 'title': 'CR Process'},
    'asset_management': {'prompt': "Track and manage project assets, such as hardware, software licenses, and intellectual property, based on the following RFP content. Include processes for asset tracking, maintenance, and disposal, and ensure compliance with any legal or regulatory requirements.\n\nRFP Content:\n{rfp_content}", 'title': 'Asset Management'},
    'qa_qc': {'prompt': "Outline the quality assurance and quality control processes that will be followed during the project, based on the following RFP content. For each process, provide detailed steps for ensuring that deliverables meet the required standards and client expectations.\n\nRFP Content:\n{rfp_content}", 'title': 'QA & QC'},
    'org_certifications': {'prompt': "List any relevant industry certifications or compliance standards that the organization adheres to, based on the following RFP content. For each certification, provide details on how it contributes to the project's success and the organization's credibility.\n\nRFP Content:\n{rfp_content}", 'title': 'Organization Certifications'}
}

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the Eligibility Criteria Extractor API'})


def deduplicate_questions(questions):
    """Remove semantically similar questions based on cosine similarity."""
    tfidf_vectorizer = TfidfVectorizer().fit_transform(questions)
    cosine_similarities = cosine_similarity(tfidf_vectorizer, tfidf_vectorizer)
    unique_questions = []
    indices_to_remove = set()
    
    for i in range(len(questions)):
        if i in indices_to_remove:
            continue
        unique_questions.append(questions[i])
        for j in range(i + 1, len(questions)):
            if cosine_similarities[i][j] > 0.8:  # Adjust threshold as needed
                indices_to_remove.add(j)
    
    return unique_questions

@app.route('/generateQuestions', methods=['POST'])
def generate_questions():
    import time  # Import time module
    
    start_time = time.time()  # Start time tracking

    rfp_id = request.form.get('rfpId', None)  # Required rfpId
    text_content = request.form.get('text_content', '')  # Optional text content

    if not rfp_id:
        return jsonify({'error': 'rfpId is required'})

    try:
        # Fetch stored chunks from Chroma database with reduced chunk size
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        document_chunks = db.similarity_search(query=rfp_id, k=10)  # Chunk size reduced to 10

        if not document_chunks:
            return jsonify({'error': 'No data found for the given rfpId'})

        all_questions = set()  # Use a set to avoid duplicates

        # Parallel processing to extract questions
        with ThreadPoolExecutor() as executor:
            generated_questions_list = list(executor.map(lambda chunk: generate_questions_from_text(chunk.page_content), document_chunks))

        # Combine all questions from parallel processing
        for generated_questions in generated_questions_list:
            all_questions.update(generated_questions)

        # If there is additional text content provided, generate questions from it
        if text_content:
            additional_questions = generate_questions_from_text(text_content)
            all_questions.update(additional_questions)

        # Convert set to list and ensure it is sorted or ordered as needed
        unique_questions = list(all_questions)

        # Ensure unique questions before numbering
        deduplicated_questions = deduplicate_questions(unique_questions)

        # Remove existing numbering if any
        deduplicated_questions_cleaned = [question.split('. ', 1)[-1].strip() for question in deduplicated_questions]

        # Limit to 20 main questions
        main_questions = deduplicated_questions_cleaned[:20]

        # Properly number questions sequentially after cleaning
        renumbered_questions = [f"{i + 1}. {question}" for i, question in enumerate(main_questions)]

        processing_time = time.time() - start_time  # Calculate processing time

        response_data = {
            'questions': renumbered_questions,
            'processing_time': f"{processing_time:.2f} seconds"  # Add processing time to the response
        }

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': 'Failed to upload data', 'details': str(e)})


@app.route('/generateSections', methods=['POST'])
def generate_sections():
    import time  # Import time module
    from concurrent.futures import ThreadPoolExecutor  # Import for parallel processing
    
    start_time = time.time()  # Start time tracking

    rfp_id = request.form.get('rfpId', '')  # Required rfpId
    section = request.form.get('section', '')  # Required section name
    prompt = request.form.get('prompt', '')  # Custom prompt is optional
    text_content = request.form.get('text_content', '')  # Optional text content

    if not rfp_id or not section:
        return jsonify({'error': 'rfpId and section are required'})

    try:
        document = collection.find_one({'_id': rfp_id})
        if not document:
            return jsonify({'error': 'No data found for the given rfpId'})

        # Join the document data content
        content = "\n".join(document['data'])

        # Limit the content size to avoid exceeding token limits
        max_content_length = 12000  # or any appropriate length to keep within token limit

        if len(content) > max_content_length:
            content = content[:max_content_length]  # Trim the content to fit within limit

        if text_content:
            content += "\n" + text_content
            # Ensure the final combined content also respects the limit
            if len(content) > max_content_length:
                content = content[:max_content_length]

        # Check if section exists in predefined sections or if it's a new one
        if section in sections:
            # Use predefined prompt
            prompt = sections[section]['prompt'].format(rfp_content=content)
        elif prompt:
            # Use the custom prompt provided by the user
            sections[section] = {'prompt': prompt, 'title': section}  # Add new section dynamically
        else:
            return jsonify({'error': 'Invalid section or missing prompt for the new section'})

        # Parallel processing for generating content
        with ThreadPoolExecutor() as executor:
            future = executor.submit(generate_content_using_prompt, content, prompt)
            output = future.result()

        processing_time = time.time() - start_time  # Calculate processing time

        response_data = {
            "outputs": [output],
            "processing_time": f"{processing_time:.2f} seconds"  # Add processing time to the response
        }

        # Update the database with generated output
        collection.update_one({'_id': rfp_id}, {'$set': {'generated_output': output}})
        logging.info(f"Generated content stored for rfpId {rfp_id}")

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': 'Failed to generate content', 'details': str(e)})


    
@app.route('/shortenText', methods=['POST'])
def shorten_text():
    content = request.form.get('content', '')

    if not content:
        return jsonify({'error': 'Content is required'})

    try:
        shortened_content = summarize_content(content)
        return jsonify({'shortened_content': shortened_content})
    except Exception as e:
        return jsonify({'error': 'Failed to shorten text', 'details': str(e)})

# New API for elaborating text
@app.route('/elaborateText', methods=['POST'])
def elaborate_text():
    content = request.form.get('content', '')

    if not content:
        return jsonify({'error': 'Content is required'})

    try:
        elaborated_content = elaborate_content(content)
        return jsonify({'elaborated_content': elaborated_content})
    except Exception as e:
        return jsonify({'error': 'Failed to elaborate text', 'details': str(e)})

def summarize_content(content):
    """Use OpenAI API to summarize the content."""
    prompt = f"Summarize the following content into a concise format:\n\n{content}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message['content'].strip()

def elaborate_content(content):
    """Use OpenAI API to elaborate the content."""
    prompt = f"Elaborate the following content with more details and explanations:\n\n{content}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message['content'].strip()

@app.route('/fetch', methods=['GET'])
def fetch_rfp_data():
    rfp_id = request.args.get('rfpId')
    if not rfp_id:
        return jsonify({'error': 'rfpId is required'})

    try:
        document = collection.find_one({'_id': rfp_id})
        if document:
            return jsonify({'rfpId': rfp_id, 'data': document['data']})
        else:
            return jsonify({'error': 'No data found for the given rfpId'})
    except Exception as e:
        return jsonify({'error': f'Failed to fetch data: {str(e)}'})

@app.route('/update', methods=['PUT'])
def update_rfp_data():
    rfp_id = request.form.get('rfpId')
    new_data = request.form.get('new_data')  # This should be the new content or questions

    if not rfp_id or not new_data:
        return jsonify({'error': 'rfpId and new_data are required'})

    try:
        # Check if rfp_id already exists
        if collection.find_one({'_id': rfp_id}):
            # Update existing entry
            collection.update_one({'_id': rfp_id}, {'$set': {'data': new_data}})
            logging.info(f"Data updated for rfpId {rfp_id}")

            # Synchronize with Chroma
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
            db.delete(ids=[rfp_id])
            document = Document(page_content=new_data, metadata={'id': rfp_id})
            add_to_chroma([document])
            return jsonify({'message': 'Data updated successfully', 'rfpId': rfp_id})
        else:
            return jsonify({'error': 'rfpId not found'})
    except Exception as e:
        return jsonify({'error': f'Failed to update data: {str(e)}'})

@app.route('/delete', methods=['DELETE'])
def delete_rfp_data():
    # Support both form-data and query parameters
    rfp_id = request.form.get('rfpId') or request.args.get('rfpId')

    if not rfp_id:
        return jsonify({'error': 'rfpId is required'})

    try:
        result = collection.delete_one({'_id': rfp_id})
        if result.deleted_count > 0:
            # Synchronize with Chroma
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
            db.delete(ids=[rfp_id])
            db.persist()
            return jsonify({'message': 'Data deleted successfully', 'rfpId': rfp_id})
        else:
            return jsonify({'error': f'Failed to delete data: rfpId {rfp_id} not found in MongoDB'})
    except Exception as e:
        return jsonify({'error': f'Failed to delete data: {str(e)}'})

@app.route('/process_s3_links', methods=['POST'])
def process_s3_links():
    s3_links = request.form.getlist('s3_links')
    rfp_id = request.form.get('rfpId', None)  # Optional rfpId
    unique_id = rfp_id if rfp_id else str(uuid.uuid4())  # Use provided rfpId or generate a unique ID

    if not s3_links:
        return jsonify({'error': 's3_links are required'})

    all_chunks = []

    for s3_link in s3_links:
        try:
            # Extract eligibility text
            eligibility_content = extract_text_from_eligibility_section_from_link(s3_link)
            # Extract full document text for all sections
            document_content = extract_full_text_from_link(s3_link)

            if eligibility_content or document_content:
                if eligibility_content:
                    eligibility_doc = Document(page_content=eligibility_content, metadata={'source': s3_link})
                    eligibility_chunks = calculate_chunk_ids([eligibility_doc])
                    all_chunks.extend(eligibility_chunks)
                    logging.info(f"Processed and chunked eligibility content from {s3_link}")
                
                if document_content:
                    document_doc = Document(page_content=document_content, metadata={'source': s3_link})
                    document_chunks = split_documents([document_doc])
                    all_chunks.extend(document_chunks)
                    logging.info(f"Processed and chunked full document content from {s3_link}")
            else:
                logging.warning(f"No content found in {s3_link}")

        except Exception as e:
            logging.error(f"Error processing S3 link {s3_link}: {str(e)}")
            return jsonify({'error': 'Failed to process S3 links', 'details': str(e)})

    if all_chunks:
        try:
            # Store chunks in Chroma and MongoDB
            add_to_chroma(all_chunks)
            if collection.find_one({'_id': unique_id}):
                collection.update_one({'_id': unique_id}, {'$set': {'data': [chunk.page_content for chunk in all_chunks]}})
                logging.info(f"Chunks updated successfully with unique ID {unique_id}")
            else:
                collection.insert_one({'_id': unique_id, 'data': [chunk.page_content for chunk in all_chunks]})
                logging.info(f"Chunks stored successfully with unique ID {unique_id}")
        except Exception as e:
            return jsonify({'error': 'Failed to store chunks', 'details': str(e)})

    return jsonify({'message': 'S3 links processed successfully', 'rfpId': unique_id})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
