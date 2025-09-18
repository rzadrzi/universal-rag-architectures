# deploy_elser.py

import os
import time
import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

print("Loading environment variables...")
load_dotenv()

es_url = os.getenv("ELASTICSEARCH_URL")
if not es_url:
    raise ValueError("ELASTICSEARCH_URL not set in .env file")

# ... (License activation part is the same) ...
print("Ensuring trial license is active...")
try:
    license_url = f"{es_url}/_license/start_trial?acknowledge=true"
    response = requests.post(license_url)
    if response.status_code == 200 and response.json().get("trial_was_started"):
        print("Trial license activated successfully.")
        print("Waiting 10 seconds for license to apply...")
        time.sleep(10)
    else:
        print("Trial license already active or could not be started. Proceeding...")
except Exception:
    print("Could not contact license endpoint. Assuming license is active.")


# --- STEP 2: CONNECT CLIENT WITH LONGER TIMEOUT ---
print("\nConnecting to Elasticsearch with a 5-minute timeout...")
# THE FIX: We add request_timeout=300 to wait up to 5 minutes.
es_client = Elasticsearch(es_url, request_timeout=300)

model_id = ".elser_model_1"

try:
    print(f"Putting trained model '{model_id}'...")
    es_client.ml.put_trained_model(model_id=model_id, input={"field_names": ["text_field"]})
    print("Model configuration put successfully.")

    print(f"Starting deployment of model '{model_id}'... This may take several minutes.")
    es_client.ml.start_trained_model_deployment(model_id=model_id, wait_for="fully_allocated")
    print("✅ ELSER model deployment is complete and the model is ready.")

except Exception as e:
    if "is already deployed" in str(e):
         print("✅ ELSER model is already deployed and running.")
    else:
        print(f"An error occurred: {e}")