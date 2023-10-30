from pdf_loader import load_pdfs

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import chainlit as cl
from langchain.chains import RetrievalQAWithSourcesChain
from typing import Dict, Optional

@cl.langchain_factory(use_async=True)
async def init():

    """
    Loads the vector data store object and the PDF documents. Creates the QA chain.
    Sets up some session variables and removes the Chainlit footer.
    
    Parameters:
    use_async (bool): Determines whether async is to be used or not.

    Returns:
    RetrievalQAWithSourcesChain: The QA chain
    """
    msg = cl.Message(content=f"Processing files. Please wait.")
    await msg.send()
    docsearch, documents = load_embeddings()
    
    humour = os.getenv("HUMOUR") == "true"
    
    chain: RetrievalQAWithSourcesChain = create_retrieval_chain(docsearch, humour=humour)
    metadatas = [d.metadata for d in documents]
    texts = [d.page_content for d in documents]
    cl.user_session.set(KEY_META_DATAS, metadatas)
    cl.user_session.set(KEY_TEXTS, texts)
    remove_footer()
    await msg.update(content=f"You can now ask questions about Onepoint HR!")#
    return chain

@cl.langchain_postprocess
async def process_response(res) -> cl.Message:
    """
    Tries to extract the sources and corresponding texts from the sources.

    Parameters:
    res (dict): A dictionary with the answer and sources provided by the LLM via LangChain.
    
    Returns:
    cl.Message: The message containing the answer and the list of sources with corresponding texts.
    """
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get(KEY_META_DATAS)
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get(KEY_TEXTS)

    found_sources = []
    if sources:
        raw_sources, file_sources = source_splitter(sources)
        for i, source in enumerate(raw_sources):
            try:
                index = all_sources.index(source)
                text = texts[index]
                source_name = file_sources[i]
                found_sources.append(source_name)
                # Create the text element referenced in the message
                logger.info(f"Found text in {source_name}")
                source_elements.append(cl.Text(content=text, name=source_name))
            except ValueError:
                continue
        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += f"\n{sources}"

    await cl.Message(content=answer, elements=source_elements).send()