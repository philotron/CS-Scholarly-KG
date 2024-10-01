from transformers import AutoTokenizer, BertAdapterModel
from setfit import SetFitModel
import gdown
import zipfile
import json
import os

# Set environment variables
with open('config.json') as json_file:
    envvar = json.load(json_file)

for key, value in envvar.items():
    os.environ[key] = value

# Download subtopic model
subtopic_tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
subtopic_model = BertAdapterModel.from_pretrained('allenai/specter2_base')
subtopic_model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

# Download topic model
try:
    topic_model = SetFitModel.from_pretrained("/app/output/", local_files_only=True)
except:
    gurl = os.getenv("setfit_gdrive", "URL")
    file = gdown.download(gurl, '/app/', quiet=False)
    
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall('/app/')