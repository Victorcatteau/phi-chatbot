

def generate_embeddings(documents: List[Document], path: Path) -> VST:
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
    except Exception as e:
        if 'docsearch' in vars() or 'docsearch' in globals():
            docsearch.persist()
        return docsearch
    return docsearch