# doc_utils.py
import io
import fitz
from typing import Union

def process_document(file_obj: Union[io.BytesIO, io.TextIOWrapper, bytes], file_type: str) -> str:
    """
    Convert the contents of the uploaded file into text.
    - Supports .txt and .pdf
    """
    if file_type == "text/plain":
        if isinstance(file_obj, bytes):
            return file_obj.decode("utf-8", errors="ignore")
        elif hasattr(file_obj, 'read'):
            file_obj.seek(0)
            text = file_obj.read()
            if isinstance(text, bytes):
                return text.decode("utf-8", errors="ignore")
            return text
        else:
            raise ValueError("Unsupported file_obj type for text/plain")

    elif file_type == "application/pdf":
        if isinstance(file_obj, bytes):
            pdf_data = file_obj
        elif hasattr(file_obj, 'read'):
            pdf_data = file_obj.read()
        else:
            raise ValueError("Unsupported file_obj type for application/pdf")

        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            all_text = []
            for page in doc:
                all_text.append(page.get_text())
        return "\n".join(all_text)

    else:
        return ""

