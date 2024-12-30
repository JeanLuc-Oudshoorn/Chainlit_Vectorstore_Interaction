# main.py
import chainlit as cl
from dotenv import load_dotenv
import aiofiles
import asyncio

# Load environment variables (OpenAI API key, etc.)
load_dotenv()

#############################################
# NEW IMPORT
#############################################
from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager

from vectorstore_utils import (
    load_faiss_vectorstore,
    add_document_to_vectorstore,
    remove_document_from_vectorstore,
    rebuild_vectorstore_from_folder,
    list_documents_in_vectorstore
)

#############################################
# GLOBAL SETTINGS
#############################################
INPUT_FOLDER = "./input_folder"
OUTPUT_FOLDER = "./output_folder"

#############################################
# CHAINLIT CONFIG
#############################################
@cl.on_chat_start
async def start():
    actions = [
        cl.Action(name="add_document", label="Add Document", description="Upload a new document", value="add_document_value"),
        cl.Action(name="remove_document", label="Remove Document", description="Remove a document from vectorstore", value="remove_document_value"),
        cl.Action(name="rebuild_vectorstore", label="Rebuild Vectorstore", description="Rebuild from input_folder", value="rebuild_vectorstore_value"),
        cl.Action(name="list_documents", label="List Documents", description="Show all docs in the vectorstore", value="list_documents_value")
    ]
    cl.user_session.set("actions", actions)
    await cl.Message(content="Welcome to the RAG Chatbot (FAISS)!", actions=actions).send()


@cl.action_callback("add_document")
async def add_document_action(action):
    files = await cl.AskFileMessage(
        content="Please upload a document to add.",
        accept=["text/plain", "application/pdf"],
        max_size_mb=20
    ).send()

    if not files:
        await cl.Message(content="No file uploaded.").send()
        return

    for f in files:
        file_path = f.path
        file_type = f.type
        file_name = f.name

        async with aiofiles.open(file_path, 'rb') as file:
            file_content = await file.read()

        add_document_to_vectorstore(file_content, file_type, file_name)
        await cl.Message(content=f"Document '{file_name}' has been added successfully.").send()


@cl.action_callback("remove_document")
async def remove_document_action(action):
    response = await cl.AskUserMessage(
        content="Enter the exact filename (e.g. 'example.pdf') to remove:"
    ).send()

    if not response or 'output' not in response or not response['output']:
        await cl.Message(content="No filename provided.").send()
        return

    doc_name = response['output'].strip()
    removed = remove_document_from_vectorstore(doc_name)
    if removed:
        await cl.Message(content=f"Document '{doc_name}' removed successfully.").send()
    else:
        await cl.Message(content=f"Document '{doc_name}' not found in the vectorstore.").send()


@cl.action_callback("rebuild_vectorstore")
async def rebuild_vectorstore_action(action):
    rebuild_vectorstore_from_folder(INPUT_FOLDER)
    await cl.Message(content=f"Vectorstore rebuilt from: {INPUT_FOLDER}").send()


@cl.action_callback("list_documents")
async def list_documents_action(action):
    docs = list_documents_in_vectorstore()
    if not docs:
        await cl.Message(content="No documents found in the vectorstore.").send()
        return

    content = "Currently stored files:\n" + "\n".join(f"- {fn}" for fn in docs)
    await cl.Message(content=content).send()


@cl.on_message
async def main_chat(message: cl.Message):
    user_input = message.content.strip()

    # Handle slash commands
    if user_input.startswith("/"):
        if user_input.startswith("/add"):
            await add_document_action(None)
            return
        elif user_input.startswith("/remove"):
            await remove_document_action(None)
            return
        elif user_input.startswith("/rebuild"):
            await rebuild_vectorstore_action(None)
            return
        elif user_input.startswith("/list"):
            await list_documents_action(None)
            return
        else:
            await cl.Message(content="Unknown command. Available commands: /add, /remove, /rebuild, /list").send()
            return

    # Normal query: do a RAG flow
    faiss_store = load_faiss_vectorstore()

    streaming_handler = AsyncIteratorCallbackHandler()
    manager = AsyncCallbackManager([streaming_handler])

    llm = ChatOpenAI(
        model='gpt-4o-mini',
        streaming=True,
        callback_manager=manager,
        temperature=0,
    )

    # Define the pre-prompt
    pre_prompt = """
You are a helpful assistant in charge of answering questions about anything.
If you do not know the answer, just say you do not know.
If there is conflicting information, please explain that there is conflicting information and give an example.

Context: {context}
Question: {question}

Provide a helpful answer:
"""
    # Construct the full prompt
    prompt = PromptTemplate(
        template=pre_prompt,
        input_variables=["context", "question"]
    )

    # Create the qa_chain with the prompt and using the FAISS vectorstore as a retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type="stuff",  # Usually “stuff” or “map_reduce”
        chain_type_kwargs={"prompt": prompt}
    )

    # We'll set up a streaming approach for the final answer
    stream_msg = cl.Message(content="")
    await stream_msg.send()

    async def process_response():
        response = ""
        async for token in streaming_handler.aiter():
            response += token
            stream_msg.content = response
            await stream_msg.update()
        return response

    chain_output_task = asyncio.create_task(qa_chain.ainvoke({"query": user_input}))
    response_task = asyncio.create_task(process_response())

    chain_output, streamed_response = await asyncio.gather(chain_output_task, response_task)

    source_docs = chain_output.get("source_documents", [])

    # Show sources
    if source_docs:
        sources_info = "\n".join(
            [f"- **{doc.metadata.get('source', 'Unknown')}**" for doc in source_docs]
        )
        sources_msg = f"**Sources:**\n{sources_info}"

        elements = []
        for doc in source_docs:
            original_bytes = doc.metadata.get("original_bytes", b"")
            src_name = doc.metadata.get("source", "unknown.pdf")
            mime_type = doc.metadata.get("mime_type", "application/pdf")

            if original_bytes:
                elements.append(
                    cl.File(
                        name=src_name,
                        content=original_bytes,
                        mime=mime_type
                    )
                )
            else:
                elements.append(
                    cl.File(
                        name=f"{src_name}.txt",
                        content=doc.page_content.encode("utf-8"),
                        mime="text/plain"
                    )
                )

        await cl.Message(content=sources_msg, elements=elements).send()
    else:
        await cl.Message(content="No sources found.").send()
