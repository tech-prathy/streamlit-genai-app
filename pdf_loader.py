# utils/pdf_loader.py
import pypdf

def load_pdf(file_file) -> str:
    """
    Extracts text chunks safely from uploaded documents.
    """
    pdf_reader = pypdf.PdfReader(file_file)
    extracted_text = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)
    return "\n".join(extracted_text)