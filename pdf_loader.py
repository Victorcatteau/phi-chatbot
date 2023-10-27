from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

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