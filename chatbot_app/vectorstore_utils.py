# vectorstore_utils.py
from langchain.docstore.document import Document


from doc_utils import process_document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from typing import List
import faiss
import os
import pickle


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_STORE_PATH = "./faiss_store.pkl"

_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def load_faiss_vectorstore() -> FAISS:
    """
    Load FAISS index from disk if it exists, otherwise create a new one.
    """
    if os.path.exists(FAISS_STORE_PATH):
        with open(FAISS_STORE_PATH, "rb") as f:
            return pickle.load(f)

    # Create a new, empty FAISS index
    dim = len(_embeddings.embed_query("test"))
    empty_index = faiss.IndexFlatL2(dim)
    empty_docstore = InMemoryDocstore({})

    # Return an empty FAISS vectorstore
    return FAISS(
        embedding_function=_embeddings,
        index=empty_index,
        docstore=empty_docstore,
        index_to_docstore_id={}
    )

def save_faiss_vectorstore(faiss_store: FAISS):
    """
    Persist FAISS index to disk via pickle.
    """
    with open(FAISS_STORE_PATH, "wb") as f:
        pickle.dump(faiss_store, f)

def add_document_to_vectorstore(file_content: bytes, file_type: str, file_name: str):
    """
    Process the file content to get text, store both the text and original bytes.
    """
    extracted_text = process_document(file_content, file_type)

    # Create a Document with both text and original bytes
    doc = Document(
        page_content=extracted_text,
        metadata={
            "source": file_name,
            "original_bytes": file_content,
            "mime_type": file_type
        }
    )

    faiss_store = load_faiss_vectorstore()

    # Unique ID for each doc
    doc_id = str(len(faiss_store.index_to_docstore_id) + 1)

    # Add to vectorstore
    faiss_store.add_documents(documents=[doc], ids=[doc_id])

    save_faiss_vectorstore(faiss_store)

def remove_document_from_vectorstore(doc_name: str) -> bool:
    """
    Remove documents from FAISS store that match 'source' == doc_name.
    Return True if at least one document was found/removed, else False.
    """
    faiss_store = load_faiss_vectorstore()

    new_texts = []
    new_metadatas = []
    document_found = False

    # Grab all docs with empty query
    all_docs = faiss_store.similarity_search("", k=9999)

    for doc in all_docs:
        if doc.metadata.get("source") == doc_name:
            document_found = True
        else:
            new_texts.append(doc.page_content)
            new_metadatas.append(doc.metadata)

    if document_found:
        # Rebuild a new store from the docs we want to keep
        new_store = FAISS.from_texts(
            new_texts,
            _embeddings,
            metadatas=new_metadatas
        )
        save_faiss_vectorstore(new_store)

    return document_found

def rebuild_vectorstore_from_folder(folder_path: str):
    """
    Remove old vectorstore file and rebuild from all files in folder.
    """
    if os.path.exists(FAISS_STORE_PATH):
        os.remove(FAISS_STORE_PATH)

    # Create a fresh store
    faiss_store = load_faiss_vectorstore()

    # Go through each file in folder
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if not os.path.isfile(full_path):
            continue

        # Simple extension-based check
        if filename.lower().endswith(".pdf"):
            file_type = "application/pdf"
        elif filename.lower().endswith(".txt"):
            file_type = "text/plain"
        else:
            continue  # skip other types

        with open(full_path, "rb") as f:
            file_content = f.read()

        extracted_text = process_document(file_content, file_type)
        doc_id = str(len(faiss_store.index_to_docstore_id) + 1)

        doc = Document(
            page_content=extracted_text,
            metadata={
                "source": filename,
                "original_bytes": file_content,
                "mime_type": file_type
            }
        )
        faiss_store.add_documents(documents=[doc], ids=[doc_id])

    save_faiss_vectorstore(faiss_store)

def list_documents_in_vectorstore() -> List[str]:
    """
    Return a list of filenames (doc.metadata["source"]) currently in the store.
    """
    faiss_store = load_faiss_vectorstore()
    all_docs = faiss_store.similarity_search("", k=9999)
    filenames = {doc.metadata.get("source", "Unknown") for doc in all_docs}
    return sorted(filenames)
