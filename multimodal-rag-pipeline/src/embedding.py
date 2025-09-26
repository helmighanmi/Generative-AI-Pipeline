"""
Author: GHNAMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

import json
import boto3
from botocore.exceptions import ClientError
from src.config import Config

config = Config()
default_model_id = config.get_embedding_model_id()
default_dim = config.get_embedding_dim()

def generate_multimodal_embeddings(prompt=None, image=None, output_embedding_length=default_dim, model_id=default_model_id):
    if not prompt and not image:
        raise ValueError("Please provide either a text prompt, base64 image, or both.")

    if not model_id:
        raise ValueError("Missing Bedrock model ID.")

    client = boto3.client(service_name="bedrock-runtime")

    body = {"embeddingConfig": {"outputEmbeddingLength": output_embedding_length}}

    if prompt:
        body["inputText"] = prompt
    if image:
        body["inputImage"] = image

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        result = json.loads(response.get("body").read())
        return result.get("embedding")

    except ClientError as err:
        print(f"[ERROR] Titan model failed: {err.response['Error']['Message']}")
        return None
