from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    """Manage the application's Chroma vector store."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def create_retriever(self, documents: List[str], k: int = 5):
        """Create a retriever from the provided documents."""
        self.vectorstore = Chroma.from_texts(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_retriever(self):
        """Get the retriever instance."""
        if self.retriever is None:
            raise ValueError("Retriever has not been created. Call create_retriever first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents based on the query."""
        if self.retriever is None:
            raise ValueError("Retriever has not been created. Call create_retriever first.")
        return self.vectorstore.similarity_search(query, k=k)
