
import json
import os

# Set environment variables
with open('/config.json') as json_file:
    envvar = json.load(json_file)

for key, value in envvar.items():
    os.environ[key] = value