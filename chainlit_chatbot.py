from typing import Dict, Optional
import os

from philosopher import Philosopher

import chainlit as cl

from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings

philosophers = ["Nietzsche", "Plato", "Descartes"]

@cl.on_chat_start
async def start():

    actions = [
        cl.Action(name = philosopher, value = philosopher, description="Chat with"+philosopher)  for philosopher in philosophers
    ]

    await cl.Message(content="Choose which philosopher you want to chat with:", actions=actions).send()

@cl.action_callback("action_button")
async def on_action(action: cl.Action):
    philosopher_name = action.name

    msg = cl.Message(content=f"Initiating philosopher. Please wait.")
    await msg.send()

    philosopher = Philosopher(name=philosopher_name)

    embeddings = OpenAIEmbeddings(show_progress_bar=True,
                              chunk_size=5)
    
    docsearch = philosopher.generate_embeddings(embeddings)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    chain = philosopher.generate_qa_chain(docsearch, memory)

    msg.content = f"`{philosopher.name}` ready. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)
    return chain

@cl.on_message
async def main(message: cl.Message):
    print(message)
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()