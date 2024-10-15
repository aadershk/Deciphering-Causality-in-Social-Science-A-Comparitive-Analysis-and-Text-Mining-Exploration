import fitz

def extract_pdf_text(pdf_path):
  with fitz.open(pdf_path) as doc:
        text = "".join([page.get_text() for page in doc])
    return text
