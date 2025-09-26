"""
Author: GHNAMI Helmi
Date: 2025-09-26
Position: Data-Science
"""
import yaml
import os

class Config:
    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_model_config(self):
        return self.config.get("model", {})

    def get_text_encoder(self):
        return self.get_model_config().get("text_encoder")

    def get_image_encoder(self):
        return self.get_model_config().get("image_encoder")

    def get_llm_provider(self):
        return self.get_model_config().get("llm_provider")

    def get_llm_model(self):
        return self.get_model_config().get("llm_model")

    def get_embedding_dim(self):
        return self.get_model_config().get("embedding_dim", 384)

    def get_bedrock_config(self):
        return self.get_model_config().get("bedrock", {})

    def get_embedding_model_id(self):
        return self.get_bedrock_config().get("embedding_model_id")

    def get_bedrock_llm_model_id(self):
        return self.get_bedrock_config().get("llm_model_id")

    def get_retriever_config(self):
        return self.config.get("retriever", {})

    def get_data_paths(self):
        return self.config.get("data", {})

    def get_pipeline_config(self):
        return self.config.get("pipeline", {})

    def get_vectorstore_config(self):
        return self.config.get("vectorstore", {})
