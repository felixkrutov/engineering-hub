import logging
import io
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook

logger = logging.getLogger(__name__)

def parse_document(file_content: bytes, mime_type: str, file_name: str) -> str:
    logger.info(f"Parsing document '{file_name}' with detected MIME type: {mime_type}")
    
    supported_mime_types = [
        'text/plain',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    ]

    if not mime_type.startswith('text/') and mime_type not in supported_mime_types:
        logger.warning(f"Unsupported file type '{mime_type}' for file '{file_name}'. Skipping parsing.")
        return f"НЕПОДДЕРЖИВАЕМЫЙ ТИП ФАЙЛА: {mime_type}"

    try:
        if mime_type == "application/pdf":
            reader = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            logger.info(f"Successfully extracted {len(text)} characters from PDF.")
            return text

        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(io.BytesIO(file_content))
            text = "\n".join([para.text for para in doc.paragraphs])
            logger.info(f"Successfully extracted {len(text)} characters from DOCX.")
            return text

        elif mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            workbook = load_workbook(filename=io.BytesIO(file_content))
            text = ""
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.value:
                            text += str(cell.value) + " "
                    text += "\n"
            logger.info(f"Successfully extracted {len(text)} characters from XLSX.")
            return text

        elif mime_type.startswith("text/"):
            text = file_content.decode('utf-8')
            logger.info(f"Successfully decoded text file with {len(text)} characters.")
            return text
            
    except Exception as e:
        logger.error(f"Failed to extract text from file '{file_name}' with MIME type {mime_type}. Error: {e}", exc_info=True)
        return f"[Error processing file: {e}]"
    
    return ""
