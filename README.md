# Conversational Exploratory Search of Scholarly Publications Using Knowledge Graphs

## Overview
This GitHub repository contains the code, models, and data resources associated with the paper: "Conversational Exploratory Search of Scholarly Publications Using Knowledge Graphs", published in ICNLSP 2024.

## Repository Structure
* **abstract-classification-service**: Includes the scripts for classifying sentences from the publication abstracts.
* **clustering-service**: Contains the code for publication text clustering and cluster naming.
* **database-dump**: Includes two data dumps from the Neo4j and Weaviate databases.
* **evaluation-questionnaire**: Provides the evaluation questionnaire, screenshots for both interfaces, and the questionnaire responses.
* **llm-api**: Contains the code for serving the large language model with corresponding endpoints for prompting.
* **neo4j-service**: Provides the scripts for setting up the Neo4j graph database with all its functionalities.
* **streamlit-app**: Includes the code for the chat interface.
* **rasa-app**: Contains all files and source code of the conversational agent built with the RASA framework.
* **topic-classification-service**: Provides the fine-tuned model and inference scripts for the topic classification using the Weaviate vector DB.

## Prompts
Listed below are the used prompts for the large language models. Dynamically inserted variables are enclosed within curly brackets. Note that Prompt 3 is not implemented and was only used for experiments for the classification of research topics.
<details>
<summary>Prompt 1: Cluster Name Generation (Zephyr-7B-Beta)</summary>
"""
Considering the themes and topics from the following TFIDF cluster tag: "{tfidf_cluster_name}", please provide a concise and descriptive name for a cluster that includes these {len(paper_list)} academic papers: \n <br> 
{paper_titles_formatted}
Respond with just the cluster name, based on the overarching themes evident in the titles and the TFIDF tag. Don't include the original TFIDF cluster tag and the word 'Cluster' in your response.
"""
</details>

<details>
<summary>Prompt 2: Comparative Text Summarization (Zephyr-7B-Beta)</summary>
Prompt 2.1 <br>
"""
Please provide a comparative analysis of the objectives of two scientific papers. <br> 
Refer the papers with their real ids: <br> 
Paper {id_a}'s objective is: {obj1} <br> 
Paper {id_b}'s objective is: {obj2} <br> 
Highlight the key differences and similarities between Paper {id_a} and Paper {id_b}. Use simple language.:
"""
<br>
<br> 
Prompt 2.2 <br>
"""
Please provide a comparative analysis of the results of two scientific papers.: <br> 
Refer the papers with their real ids: <br> 
Results of Paper {id_a}: {res1} <br> 
Results of Paper {id_b}: {res2} <br> 
Highlight the key differences and similarities between Paper {id_a} and Paper {id_b}. Use simple language.:
"""
<br> 
<br>
Prompt 2.3 <br>
"""
Please provide a comparative analysis of the TLDR of two scientific papers.: <br> 
TLDR of Paper {id_a}: {tldr1} <br> 
TLDR of Paper {id_b}: {tldr2} <br> 
Highlight the key differences and similarities between Paper {id_a} and Paper {id_b}. Use simple language.:
"""
</details>

<details>
<summary>Prompt 3: LLM-Based Research Topic Classification (GPT-3.5-Turbo)</summary>
"""
You are supposed to classify a query into one of the topics provided. These topics are various fields of NLP. Your answer should be in the following format: *topic name*. <br> 
Nothing else should be included in the output. <br> 
Make sure there is no extra punctuation including full stops, quotation marks or anything of that sort. You are supposed to EXACTLY use the topics from the list provided.
If you think it is a random question and not in the field of NLP, then return the topic as 'none'. <br> 
You can only provide your answer from the following topics and the topics are: 
Multimodality <br> 
Natural Language Interfaces <br> 
Semantic Text Processing <br> 
Semantic Analysis <br> 
Syntactic Text Processing <br> 
Linguistic and Cognitive NLP <br> 
Responsible NLP <br> 
Reasoning <br> 
Multilinguality <br> 
Information Retrieval <br> 
Information Extraction and Text Mining <br> 
Text Generation <br> 
Query: {query}. <br> 
Topic:  
"""
</details>

## Interface Screenshots
Below are screenshots showing a visual side-by-side comparison of the conversational and graphical interfaces from the human evaluation study.
<table>
  <tr>
    <td style="text-align: center;">
      <h4>Conversational Interface</h4>
      <img src="evaluation-questionnaire/interface-screenshots/conversational_interface_screenshot.png" alt="Conversational Interface" style="width: 500px; height: 300px;">
    </td>
    <td style="text-align: center;">
      <h4>Graphical Interface</h4>
      <img src="evaluation-questionnaire/interface-screenshots/graphical_interface_screenshot.png" alt="Graphical Interface" style="width: 500px; height: 300px;">
    </td>
  </tr>
</table>
