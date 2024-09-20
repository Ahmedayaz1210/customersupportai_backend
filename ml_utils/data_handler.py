"""
data_handler.py

This module provides utility functions for handling data, such as fetching and processing documents from URLs.
"""

import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# Initialize the text splitter with specific parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

def get_docs(urls):
    """
    Fetches and processes documents from the given list of URLs.

    This function sends HTTP GET requests to the provided URLs, retrieves the HTML content,
    extracts the text, and splits the text into smaller chunks using a text splitter.

    Args:
        urls (list of str): A list of URLs to fetch documents from.

    Returns:
        list of Document: A list of Document objects containing the text chunks and metadata.

    Raises:
        Exception: If the HTTP request to any URL fails.
    """
    documents = []
    for url in urls:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            html_content = response.text
        else:
            raise Exception(f"Failed to retrieve the webpage: {url}")
        
        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(html_content, 'html.parser')
        # Extract text from the HTML (you can customize this to extract specific tags)
        text = soup.get_text(strip=True)
        # Convert text to langchain Document
        docs = Document(page_content=text, metadata={"source": url})
        documents.append(docs)
    
    # Split the documents into chunks
    doc_chunks = text_splitter.split_documents(documents)
    
    return doc_chunks

if __name__ == "__main__":
    """
    Main entry point for testing the get_docs function.

    This block will execute when the module is run as a script. It fetches documents from a predefined
    list of URLs and prints the content and length of the first document chunk.
    """
    docs = get_docs(["https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.chromium.AsyncChromiumLoader.html"])
    print(docs[0].page_content)
    print(len(docs))