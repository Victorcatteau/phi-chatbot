from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain.schema import Document
from typing import TypeVar, List

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from config import cfg

VST = TypeVar("VST", bound="VectorStore")

data_path = "../ml/data/pdf"

def load_pdfs(path: Path) -> List[Document]:
    """
    Loads the PDFs and extracts a document per page.
    The page details are added to the extracted metadata
    
    Parameters:
    path (Path): The path where the PDFs are saved.
    
    Returns:
    List[Document]: Returns a list of values
    """
    assert path.exists()
    all_pages = []
    for pdf in path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf.absolute()))
        pages: List[Document] = loader.load_and_split()
        #for i, p in enumerate(pages):
        #    file_name = re.sub(r".+[\\/]", '', p.metadata['source'])
        #    p.metadata['source'] = f"{file_name} page {i + 1}"
        all_pages.extend(pages)
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=2000,
            chunk_overlap=0,
        )
        docs = text_splitter.split_documents(pages)
    return docs

def generate_embeddings(
    documents: List[Document], path: Path, faiss_persist_directory: str
) -> VST:
    """
    Receives a list of documents and generates the embeddings via OpenAI API.

    Parameters:
    documents (List[Document]): The document list with one page per document.
    path (Path): The path where the documents are found.

    Returns:
    VST: Recturs a reference to the vector store.
    """
    try:
        docsearch = FAISS.from_documents(documents, cfg.embeddings)
        docsearch.save_local(cfg.faiss_persist_directory)
        logger.info("Vector database persisted")
    except Exception as e:
        logger.error(f"Failed to process {path}: {str(e)}")
        if 'docsearch' in vars() or 'docsearch' in globals():
            docsearch.persist()
        return docsearch
    return docsearch
