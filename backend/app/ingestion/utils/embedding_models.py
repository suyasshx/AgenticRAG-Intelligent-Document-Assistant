from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import logger


def get_embedding_model():
    """
    Return the local Hugging Face embedding model.
    """

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    logger.info(f"Loaded embedding model: {model_name}")

    return embeddings