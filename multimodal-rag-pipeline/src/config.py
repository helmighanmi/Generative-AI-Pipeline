"""
Author: GHANMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

import os
import yaml


class Config:
    """
    Loads and provides access to configuration values
    from `config/config.yaml`.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        # Resolve absolute path (works if called from anywhere in repo)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, config_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Config file not found at: {full_path}")

        with open(full_path, "r") as f:
            self.config = yaml.safe_load(f)

    # ---------------------------
    # Embeddings
    # ---------------------------
    def get_embedding_provider(self) -> str:
        return self.config["model"]["embedding"]["provider"]

    def get_embedding_model_id(self) -> str:
        return self.config["model"]["embedding"]["model_id"]

    def get_embedding_dim(self) -> int:
        return self.config["model"]["embedding"].get("dim", 384)

    def get_text_encoder(self) -> str:
        return self.config["model"]["embedding"].get("model_id")

    def get_image_encoder(self) -> str:
        return self.config["model"].get("image_encoder")

    # ---------------------------
    # LLM
    # ---------------------------
    def get_llm_provider(self) -> str:
        return self.config["model"]["llm"]["provider"]

    def get_llm_model_id(self) -> str:
        return self.config["model"]["llm"]["model_id"]

    def get_llm_params(self) -> dict:
        return {
            "temperature": self.config["model"]["llm"].get("temperature", 0.7),
            "max_tokens": self.config["model"]["llm"].get("max_tokens", 512),
        }

    # ---------------------------
    # Retrieval & Pipeline
    # ---------------------------
    def get_retriever_config(self) -> dict:
        return self.config["retriever"]

    def get_pipeline_config(self) -> dict:
        return self.config["pipeline"]

    # ---------------------------
    # Data paths
    # ---------------------------
    def get_data_paths(self) -> dict:
        return self.config["data"]

    # ---------------------------
    # Vector store
    # ---------------------------
    def get_vectorstore_config(self) -> dict:
        return self.config["vectorstore"]
