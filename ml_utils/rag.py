from ml_utils.doc_embedding import get_langchain_embeddings
from ml_utils.data_handler import get_docs


class RAG:
    """
    The RAG (Retrieval-Augmented Generation) class provides methods to manage a vector store
    for document embeddings and perform similarity searches.

    Attributes:
        db_filepath (str): The file path to the database for persisting the vector store.
        collection_name (str): The name of the collection in the vector store.
        thresh (float): The threshold for relevance scores in search results.
        urls (list): A list of URLs added to the vector store.
        vector_store (Chroma): The vector store instance for managing document embeddings.
    """

    def __init__(self, db_filepath, collection_name, thresh=0.2) -> None:
        """
        Initializes the RAG instance with a vector store.

        Args:
            db_filepath (str): The file path to the database for persisting the vector store.
            collection_name (str): The name of the collection in the vector store.
            thresh (float, optional): The threshold for relevance scores in search results. Defaults to 0.2.
        """
        self.urls = []

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=get_langchain_embeddings(),
            persist_directory=db_filepath
        )
        self.thresh = thresh

    def add_url(self, url):
        """
        Adds a document from the given URL to the vector store if it does not already exist.

        Args:
            url (str): The URL of the document to be added.
        """
        doc_splits = get_docs([url])
        # Check if the URL already exists in the vector store
        db_data = self.vector_store.get(include=['metadatas'])
        for metadata in db_data['metadatas']:
            if metadata['source'] == doc_splits[0].metadata['source']:
                print(f"Skipping {doc_splits[0].metadata['source']} as it already exists in the vector store")
                return

        self.vector_store.add_documents(doc_splits)
        self.urls.append(url)

    def search(self, query, top_k=3):
        """
        Searches the vector store for documents similar to the given query.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of top results to return. Defaults to 3.

        Returns:
            list: A list of documents that have a relevance score above the threshold.
        """
        docs_db = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        results = []
        for doc, score in docs_db:
            if score > self.thresh:
                results.append(doc)

        return results

    def get_context(self, query):
        """
        Retrieves the most relevant document for the given query.

        Args:
            query (str): The search query.

        Returns:
            str: The content of the most relevant document, or an empty string if no relevant document is found.
        """
        docs = self.search(query, top_k=1)
        if len(docs) == 0:
            return ""
        return docs[0].page_content