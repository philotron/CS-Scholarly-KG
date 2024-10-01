FROM python:3.8

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY . /

# Download the models and set the environment variables 
RUN python app/download_models.py

# Expose the necessary ports for flask
EXPOSE 5000

# Run the flask app
ENTRYPOINT [ "python" ]
CMD [ "app/flask_app.py" ]