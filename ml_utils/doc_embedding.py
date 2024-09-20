"""
doc_embedding.py

This module provides functions to embed documents using the Hugging Face embedding model (all-MiniLM-L6-v2).
The embeddings can be used for tasks such as reranking documents after query search and retrieval.
For reranking, the Jina model (jina-reranker-v1-turbo-en) from Hugging Face is used.

References:
- Embedding Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Reranking Model: https://huggingface.co/jinaai/jina-reranker-v1-turbo-en
"""

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings(texts, model="all-MiniLM-L6-v2"):
    """
    Generates embeddings for a list of texts using the specified Hugging Face model.

    Args:
        texts (list of str): A list of texts to be embedded.
        model (str, optional): The name of the Hugging Face model to use. Defaults to "all-MiniLM-L6-v2".

    Returns:
        numpy.ndarray: An array of embeddings for the input texts.
    """
    model = SentenceTransformer(f"sentence-transformers/{model}")  # or any other pretrained model
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def get_langchain_embeddings(model="all-MiniLM-L6-v2"):
    """
    Provides embeddings compatible with the LangChain vector database.

    Args:
        model (str, optional): The name of the Hugging Face model to use. Defaults to "all-MiniLM-L6-v2".

    Returns:
        HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings initialized with the specified model.
    
    References:
        https://api.python.langchain.com/en/latest/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html
    """
    model_name = f"sentence-transformers/{model}"
    return HuggingFaceEmbeddings(model_name=model_name)

if __name__ == "__main__":
    """
    Main entry point for testing the embedding functions.

    This block will execute when the module is run as a script. It generates embeddings for a predefined
    list of example texts and prints the type and shape of the first embedding.
    """
    texts = ["This is an example sentence", "Each sentence is converted"]
    embeddings = get_embeddings(texts)
    print(type(embeddings[0]), embeddings[0].shape)