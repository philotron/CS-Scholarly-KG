from flask import Flask, request, jsonify
import torch
from transformers import pipeline
import logging
import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

model = "zephyr-7b-beta"
fine_tuned_model = "fine-tune"


prompt = "Once upon a time"

app = Flask(__name__)



def generate_response(message, model):
    messages = [
        {
            "role": "system",
            "content": "You are a chatbot who always responds in academic language and is academically smart. Keep your answers short and answer with maximum 2 sentences.",
        
            ## Add User assistant example
        },
        {
            "role": "user", 
            "content": message,
        }
    ]


    # create a chat completion
    completion = openai.chat.completions.create(
    model=model,
    messages=messages
    )

    return completion.choices[0].message.content


# Set the desired logging level
app.logger.setLevel(logging.DEBUG)  


# Function to extract the last response from the LLM's output and clean it
def extract_last_response(full_text, last_user_message):
    # Find the start of the last response
    start_idx = full_text.rfind(last_user_message) + len(last_user_message)
    # Extract the last response
    last_response = full_text[start_idx:].strip()
    # Remove the </s> token
    last_response = last_response.replace('</s>', '').strip()
    # Remove <|assistant|> 
    last_response = last_response.replace('<|assistant|>', '').strip()

    return last_response


@app.route('/generate_comparison/objective', methods=['POST'])
def generate_comparison_objective():
    try:
        print("Request to comparison")

        app.logger.debug("This is a debug message")

        app.logger.debug(f"This is the request {request}")

        data = request.json

        if not data:
            return jsonify({'error': 'Invalid request, missing data'}), 400
        
        obj1 = data["paper_1_obj"]
        obj2 = data["paper_2_obj"]
        id_a = data["id_a"]
        id_b = data["id_b"]

        # Construct the prompt for comparison
        comparison_prompt_objective = f"""\
            Please provide a comparative analysis of the objectives of two scientific papers. \
            Refer the papers with their real ids:\
            Paper {id_a}'s objective is: {obj1}\
            Paper {id_b}'s objective is: {obj2}\
            Highlight the key differences and similarities between Paper {id_a} and Paper {id_b}. Use simple language.:\
            """
        
        # Generate responses for each prompt separately
        response_to_objective = generate_response(comparison_prompt_objective, model)
        
        return jsonify({'comparison': response_to_objective})
    except Exception as e:
        app.logger.debug(f"This is the error {str(e)}")

        return jsonify({'error': str(e)}), 500
    

@app.route('/generate_comparison/results', methods=['POST'])
def generate_comparison_results():
    try:
        print("Request to comparison")

        app.logger.debug("This is a debug message")

        app.logger.debug(f"This is the request {request}")

        data = request.json

        if not data:
            return jsonify({'error': 'Invalid request, missing data'}), 400
        
        res1 = data["paper_1_res"]
        res2 = data["paper_2_res"]
        id_a = data["id_a"]
        id_b = data["id_b"]

        comparison_prompt_results = f"""\
            Please provide a comparative analysis of the results of two scientific papers.:\
            Refer the papers with their real ids:\
            Results of Paper {id_a}: {res1}\
            Results of Paper {id_b}: {res2}\
            Highlight the key differences and similarities between Paper {id_a} and Paper {id_b}. Use simple language.:\
            """

        # Generate responses for each prompt separately
        response_to_results = generate_response(comparison_prompt_results, model)

        return jsonify({'comparison': response_to_results})
    except Exception as e:
        app.logger.debug(f"This is the error {str(e)}")

        return jsonify({'error': str(e)}), 500
    

@app.route('/generate_comparison/tldr', methods=['POST'])
def generate_comparison_tldr():
    try:
        print("Request to comparison")

        app.logger.debug("This is a debug message")

        app.logger.debug(f"This is the request {request}")

        data = request.json

        if not data:
            return jsonify({'error': 'Invalid request, missing data'}), 400

        tldr1 = data["paper_1_tldr"]
        tldr2 = data["paper_2_tldr"]
        id_a = data["id_a"]
        id_b = data["id_b"]

        comparison_prompt_tldr = f"""\
            Please provide a comparative analysis of the TLDR of two scientific papers.:\
            TLDR of Paper {id_a}: {tldr1}\
            TLDR of Paper {id_b}: {tldr2}\
            Highlight the key differences and similarities between Paper {id_a} and Paper {id_b}. Use simple language.:\
            """
        
        # Generate responses for each prompt separately
        response_to_tldr = generate_response(comparison_prompt_tldr, model)

        return jsonify({'comparison' : response_to_tldr})
    except Exception as e:
        app.logger.debug(f"This is the error {str(e)}")

        return jsonify({'error': str(e)}), 500
    

@app.route('/generate_cluster_name', methods=['POST'])
def generate_cluster_name():
    try:
        print("Request to cluster name")

        data = request.json

        app.logger.debug(f"This is the request data {data}")

        if not data:
            return jsonify({'error': 'Invalid request, missing data'}), 400

        
        tfidf_cluster_name = data.get("tfidf_cluster_name")
        paper_list = data.get("paper_titles")  # Assuming paper_list is a list of paper titles

        # Ensure there is a list of papers and it's not empty
        if not paper_list or not isinstance(paper_list, list):
            return jsonify({'error': 'Invalid request, missing paper list or paper list is not a list'}), 400

        paper_titles_formatted = '\n'.join([f"Title {i+1}: {title}" for i, title in enumerate(paper_list)])

        # Prompt to generate a cluster name
        cluster_name_prompt = f"""
        Considering the themes and topics from the following TFIDF cluster tag: "{tfidf_cluster_name}", 
        please provide a concise and descriptive name for a cluster that includes these {len(paper_list)} academic papers:
        {paper_titles_formatted}
        Respond with just the cluster name, based on the overarching themes evident in the titles and the TFIDF tag. Don't include the original TFIDF cluster tag and the word 'Cluster' in your response.
        """

        # Generate responses for each prompt separately
        response_to_cluster_name = generate_response(cluster_name_prompt, model)

        app.logger.debug(f"This is the cluster name {response_to_cluster_name}")

        return jsonify({'cluster_name' : response_to_cluster_name})
    except Exception as e:
        app.logger.debug(f"This is the error {str(e)}")

        return jsonify({'error': str(e)}), 500    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
