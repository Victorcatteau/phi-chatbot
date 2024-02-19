import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from pathlib import Path
from pdf_loader import load_pdfs

from langchain.vectorstores import FAISS
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = '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'

prompt_template = """Let's think step by step. Use the tone and information of the following pieces of context to answer the question as if the author was answering the question.

{context}

Question: {question}
Answer:"""

DEFAULT_TEMPLATE = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

data_path = "data/"

class Philosopher(BaseModel):
    name: str
    k: int = 4
    template: PromptTemplate = DEFAULT_TEMPLATE
    temperature: float = 1

    def generate_embeddings(self, embeddings):
        embeddings_path = Path(data_path + "embeddings/"+self.name)
        if not os.path.exists(embeddings_path):
            raw_data_path = Path(data_path + "raw/"+self.name)
            docs = load_pdfs(raw_data_path)
            docsearch = FAISS.from_documents(docs, embeddings)
            docsearch.save_local(embeddings_path)
        else:
            docsearch = FAISS.load_local(embeddings_path, embeddings)
        return docsearch

    def generate_qa_chain(self, docsearch, memory):
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model_name="gpt-3.5-turbo", temperature=self.temperature, streaming=True),
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={'k': self.k}),
            memory=memory,
            return_source_documents=True,
        )
        return chain
