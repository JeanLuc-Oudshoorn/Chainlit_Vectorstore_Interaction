# RAG Chatbot with built-in vectorstore management

This repository contains a Retrieval-Augmented Generation (RAG) chatbot built using FAISS, Chainlit, and OpenAI's GPT models. The chatbot allows for document-based querying, enabling the user to upload files, retrieve answers, and download source documents. It supports chunking for larger documents and features tools for managing the vectorstore, including adding, listing, removing, and rebuilding documents.

## Features

### Chatbot Functionality
- **Query Answers**: Ask questions and get answers based on uploaded documents.
- **Source Documents**: Download the original source files used to generate the answer.
- **Streaming Responses**: See the answer token by token as it is generated.

### Vectorstore Management
- **Add Documents**: Upload `.pdf` or `.txt` files to the vectorstore.
- **Remove Documents**: Delete documents by filename.
- **List Documents**: View all documents currently in the vectorstore.
- **Rebuild Vectorstore**: Recreate the vectorstore from all files in a specified folder.

### Document Handling
- **Original File Storage**: Retains the original file bytes for download.
- **Chunking Strategy**: Uses a `RecursiveCharacterTextSplitter` to split larger documents into manageable chunks for embedding and retrieval.
- **Metadata Storage**: Stores file metadata like filename, MIME type, and original content for better document management.

### Customization
- **Model Configuration**: Uses the `chatgpt-4o-mini` model by default (configurable).
- **Environment Variables**: API keys and other sensitive data are loaded via a `.env` file.

## Installation

### Prerequisites
- Python 3.8+
- A valid OpenAI API key

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file in the root directory with the following content:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Run the Application
```bash
chainlit run main.py
```

### Open in Browser
The application will be available at: [http://localhost:8000](http://localhost:8000)

## Usage

### Adding Documents
1. Click the **Add Document** button or type `/add` in the chat.
2. Upload a `.pdf` or `.txt` file.
3. The document will be processed and added to the vectorstore.

### Querying
1. Type a question in the chat.
2. The chatbot will retrieve relevant information from the documents and provide an answer.
3. Download the source documents from the "Sources" section if needed.

### Listing Documents
1. Click the **List Documents** button or type `/list` in the chat.
2. See a list of all filenames currently stored in the vectorstore.

### Removing Documents
1. Click the **Remove Document** button or type `/remove` in the chat.
2. Provide the exact filename to remove.
3. The document and all its chunks will be removed from the vectorstore.

### Rebuilding the Vectorstore
1. Click the **Rebuild Vectorstore** button or type `/rebuild` in the chat.
2. The vectorstore will be rebuilt from all files in the `input_folder` directory.

## File Structure

```
├── main.py                 # Main application logic
├── vectorstore_utils.py    # FAISS and document management utilities
├── doc_utils.py            # Document processing functions (e.g., text extraction)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── input_folder/           # Folder for rebuilding the vectorstore
├── output_folder/          # Optional folder for logs or exports
├── faiss_store.pkl         # Persisted FAISS vectorstore (generated at runtime)
```

## Configuration

### Model Configuration
The chatbot uses the `chatgpt-4o-mini` model by default. To change the model, update the `model_name` parameter in `main.py`:
```python
llm = ChatOpenAI(
    model_name="your_model_name_here",
    streaming=True,
    callback_manager=manager,
    temperature=0,
)
```

### Chunking Parameters
You can adjust chunk size and overlap in `vectorstore_utils.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

## Future Enhancements
- Add support for more file formats (e.g., Word documents).
- Improve UI/UX for document management.
- Add authentication for secure access.

## Contributing
Feel free to submit issues and pull requests to improve the project. Contributions are welcome!

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Enjoy using the RAG Chatbot with FAISS and Chainlit!
