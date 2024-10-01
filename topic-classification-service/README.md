### What is here
This docker container downloads and prepares both setfit and similarity embedding models that predict the main topic and subtopics.  
The container runs a Flask API that takes in a user query and returns the predicted topic and subtopic.

### Config.json
1. weaviate_uri: URL where the weaviate database is running
2. flask_port: Port for the flask app
3. setfit_gdrive: Google drive link where the trained setfit model is located for inference purposes (it is provided in /app/output)