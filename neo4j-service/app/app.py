from flask import Flask, jsonify, request
from Neo4jApp import Neo4jApp
import os
import pandas as pd

app = Flask(__name__)
neo4j_app = Neo4jApp()

"""
This Flask application provides a set of RESTful API endpoints to interact with a Neo4j database. 
The application defines multiple routes to retrieve information related to topics, their parent and child nodes, 
NLP taxonomy, paper details, clusters, and subclusters.
"""

@app.route('/api', methods=['GET'])
def home():
    return jsonify({"message": "Hello, World!"})

@app.route('/get_children_nodes', methods=['GET', 'POST'])
def get_children_nodes():
    topic_name = request.form.get('topic_name', request.form.get('topic_name', None))

    if topic_name is not None:
        return jsonify(neo4j_app.get_children_nodes(topic_name))

    return jsonify({'error': "Topic name is missing"}), 401

@app.route('/get_parent_nodes', methods=['GET', 'POST'])
def get_parent_nodes():
    topic_name = request.form.get('topic_name', request.form.get('topic_name', None))

    if topic_name is not None:
        return jsonify(neo4j_app.get_parent_nodes(topic_name))

    return jsonify({'error': "Topic name is missing"}), 401

@app.route('/get_topic_definition', methods=['GET', 'POST'])
def get_topic_definition():
    topic_name = request.form.get('topic_name', request.form.get('topic_name', None))

    if topic_name is not None:
        return jsonify(neo4j_app.get_topic_definition(topic_name))

    return jsonify({'error': "Topic name is missing"}), 401

@app.route('/get_nlp_taxonomy', methods=['GET', 'POST'])
def get_nlp_taxonomy():
    return jsonify(neo4j_app.get_nlp_taxonomy())

@app.route('/get_paper_details', methods=['GET', 'POST'])
def get_paper_details():
    paper_id = request.form.get('paper_id', request.form.get('paper_id', None))

    if paper_id is not None:
        return jsonify(neo4j_app.get_paper_details(int(paper_id)))
    
    return jsonify({'error': "Paper ID is missing"}), 401


@app.route('/get_clusters_by_topic', methods=['GET'])
def get_clusters_by_topic():
    topic_name = request.form.get('topic_name', request.form.get('topic_name', None))

    if topic_name is not None:
        return jsonify(neo4j_app.get_clusters_by_topic(topic_name))

    return jsonify({'error': "Topic name is missing"}), 401

@app.route('/get_subclusters_by_cluster', methods=['GET'])
def get_subclusters_by_cluster():
    cluster_id = request.form.get('cluster_id', request.form.get('cluster_id', None))

    if cluster_id is not None:
        return jsonify(neo4j_app.get_subclusters_by_cluster_id(cluster_id))

    return jsonify({'error': "Cluster name is missing"}), 401

@app.route('/get_publications_by_cluster_and_subclusters', methods=['GET'])
def get_publications_by_cluster_and_subclusters():
    cluster_id = request.form.get('cluster_id', request.form.get('cluster_id', None))

    if cluster_id is not None:
        return jsonify(neo4j_app.get_publications_by_cluster_and_subclusters(cluster_id))

    return jsonify({'error': "Cluster is missing"}), 401

if __name__ == '__main__':
    port = os.getenv("flask_port", 5001)
    app.run(debug=True, host='0.0.0.0', port=port)