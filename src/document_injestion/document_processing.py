from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessing:
    """Handles loading and splitting documents from supported sources."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum size of a text chunk.
            chunk_overlap: Number of characters shared by adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_character_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load a document from a URL."""
        loader = WebBaseLoader(url)
        return loader.load()

    @staticmethod
    def _get_valid_path(
        path: Union[str, Path],
        *,
        directory: bool = False,
    ) -> Path:
        """Convert a source to a Path and validate its type."""
        resolved_path = Path(path)

        if directory:
            if not resolved_path.is_dir():
                raise ValueError(f"{path} is not a directory")
        elif not resolved_path.is_file():
            raise ValueError(f"{path} is not a file")

        return resolved_path

    def load_pdf_from_dir(self, dir_path: Union[str, Path]) -> List[Document]:
        """Load all PDF documents from a directory."""
        path = self._get_valid_path(dir_path, directory=True)

        loader = PyPDFDirectoryLoader(str(path))
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a document from an individual PDF file."""
        path = self._get_valid_path(file_path)

        loader = PyPDFLoader(str(path))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a text document from a UTF-8 text file."""
        path = self._get_valid_path(file_path)

        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF files, PDF directories, or text files.

        Args:
            sources: URLs or local file/directory paths.

        Returns:
            A list of all loaded LangChain documents.
        """
        documents: List[Document] = []

        for source in sources:
            if source.startswith(("http://", "https://")):
                documents.extend(self.load_from_url(source))
                continue

            path = Path(source)

            if path.is_dir():
                documents.extend(self.load_pdf_from_dir(path))
            elif path.is_file() and path.suffix.lower() == ".pdf":
                documents.extend(self.load_from_pdf(path))
            elif path.is_file() and path.suffix.lower() == ".txt":
                documents.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"{source} is not a supported file or directory. "
                    "Use a URL, PDF file, text file, or PDF directory."
                )

        return documents
