# check_elser_status.py

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

print("Loading environment variables...")
load_dotenv()

es_url = os.getenv("ELASTICSEARCH_URL")
if not es_url:
    raise ValueError("ELASTICSEARCH_URL not set in .env file")

print("Connecting to Elasticsearch...")
es_client = Elasticsearch(es_url)

print("Checking ELSER model status...")
try:
    status = es_client.ml.get_trained_models(model_id=".elser_model_1")
    
    # Extract and print the deployment state
    deployment_state = status['trained_model_configs'][0]['deployment_state']
    print("\n--- ELSER Model Status ---")
    print(f"State: {deployment_state}")
    print("--------------------------")
    
    if deployment_state == 'started':
        print("\n✅ Success! The ELSER model is deployed and ready.")
    else:
        print("\n⏳ In progress. The model is not fully deployed yet. Please wait a few more minutes and run this script again.")

except Exception as e:
    print(f"\n❌ An error occurred or the model was not found: {e}")
    print("Please ensure you have run 'deploy_elser.py' and waited 3-5 minutes.")