from langchain.prompts import PromptTemplate

prompt_template = """Let's think step by step. Use the tone and information of the following pieces of context to answer the question as if the author was answering the question.

{context}

Question: {question}
Answer:"""

DEFAULT_TEMPLATE = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)
