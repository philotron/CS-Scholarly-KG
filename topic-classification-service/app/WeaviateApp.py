import os
import weaviate
from weaviate.exceptions import UnexpectedStatusCodeException

class WeaviateApp:

    def __init__(self):
        #uri = os.getenv("uri_weaviate", "http://0.0.0.0:8080")
        uri = os.getenv("uri_weaviate", "http://localhost:8081")
        self.driver = weaviate.Client(uri)

    def close(self):
        self.driver.close()


    def get_n_closest_subtopics(self, embedding, topic, n=20):
        print("SCHEMA", self.driver.schema.get())

        data = self.driver.query.get("Publication", ["subtopic", "neo4jID"]) \
                            .with_where({"path": "topic", "operator": "Equal", "valueString": topic}) \
                            .with_near_vector({"vector": embedding}) \
                            .with_additional(["distance"]).with_limit(n).do()
        
        new_data = self.driver.query.get("Publication", ["subtopic", "neo4jID", "topic"]) \
            .with_limit(100).do()  # Adjust the limit as needed

        print("DATA", data)

        print("NEW DATA ", new_data['data']['Get']['Publication'])
        
        return data['data']['Get']['Publication']