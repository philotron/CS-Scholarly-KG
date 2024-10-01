from transformers import AutoTokenizer, BertAdapterModel
from setfit import SetFitModel
from WeaviateApp import WeaviateApp

class TopicPredictor:
    topic_for_label = {0:'Multimodality',
                    1:'Natural Language Interfaces',
                    2:'Semantic Text Processing',
                    3:'Sentiment Analysis',
                    4:'Syntactic Text Processing',
                    5:'Linguistics & Cognitive NLP',
                    6:'Responsible NLP',
                    7:'Reasoning',
                    8:'Multilinguality',
                    9:'Information Retrieval',
                    10:'Information Extraction & Text Mining',
                    11:'Text Generation',
                   -1:'Random Question'}
                    
    
    def __init__(self):
        # Connect to Weaviate
        self.weaviate_client = WeaviateApp()

        # Load subtopic model and tokenizer
        self.subtopic_tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        self.subtopic_model = BertAdapterModel.from_pretrained('allenai/specter2_base')
        self.subtopic_model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

        # Load topic model
        self.topic_model = SetFitModel.from_pretrained("output", local_files_only=True)

    def predict(self, query):
        topic = self.predict_topic(query)
        if topic == "Random Question":
            return topic, 'random'
        subtopic = self.predict_subtopic(query, topic)

        return topic, subtopic
    
    def predict_topic(self, query):
        topic_label = int(self.topic_model([query]))
        return self.topic_for_label[topic_label]
    
    def predict_subtopic(self, query, topic):

        embedding = self.embed_user_query(query)

        publications = self.weaviate_client.get_n_closest_subtopics(embedding, topic)

        # Weighted KNN on the nearest publications
        topics = {}
        for record in publications:
            field = record["subtopic"]
            
            # Calculate weight using the distance
            weight = 1/(record["_additional"]["distance"] **2)

            topics[field] = topics.get(field, []) + [weight]

        # Sum the weights and get the maximum
        classified_topic = 'random'
        cumulative_similarity = 0
        for key, value in topics.items():
            label = key
            cumul = sum(value)
            if cumulative_similarity < cumul:
                classified_topic = label
                cumulative_similarity = cumul

        return classified_topic
    
    
    def embed_user_query(self, query):
        inputs = self.subtopic_tokenizer([query], padding=True, truncation=True,
                                return_tensors="pt", return_token_type_ids=False, max_length=512)

        output = self.subtopic_model(**inputs)
        # take the first token in the batch as the embedding
        embedding = output.last_hidden_state[:, 0, :]
        embedding = embedding.tolist()[0]
        return embedding

