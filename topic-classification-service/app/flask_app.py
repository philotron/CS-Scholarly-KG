from flask import Flask, jsonify, request
from TopicPredictor import TopicPredictor
import os

app = Flask(__name__)
topic_predictor = TopicPredictor()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    query = request.form.get('query', request.form.get('query', None))
    app.logger.info(f"Query: {str(query)}")
    if query is not None:
        try:
            topic, subtopic = topic_predictor.predict(query)
            
            return jsonify({'topic': topic, 'subtopic': subtopic})
        except Exception as e:
            app.logger.info(str(e))
        

    return jsonify({'error': "Query is missing"}), 401

if __name__ == '__main__':
    port = os.getenv("flask_port", 5500)
    app.run(debug=True, host='0.0.0.0', port=port)