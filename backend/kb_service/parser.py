import logging
import io
import rarfile
import magic
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook

logger = logging.getLogger(__name__)

def parse_document(file_name: str, file_content: bytes, mime_type: str) -> str:
    logger.info(f"Parsing document '{file_name}' with detected MIME type: {mime_type}")

    SUPPORTED_TYPES = [
        'text/plain',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/x-rar-compressed',
        'application/x-rar',
    ]

    if not mime_type.startswith('text/') and mime_type not in SUPPORTED_TYPES:
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

        elif mime_type in ['application/x-rar-compressed', 'application/x-rar']:
            logger.info(f"Processing RAR archive: {file_name}")
            all_texts = []
            try:
                with rarfile.RarFile(io.BytesIO(file_content)) as rf:
                    for info in rf.infolist():
                        if info.is_dir():
                            continue
                        
                        inner_file_content = rf.read(info)
                        inner_mime_type = magic.from_buffer(inner_file_content, mime=True)
                        inner_text = parse_document(info.filename, inner_file_content, inner_mime_type)
                        
                        if not inner_text.startswith("НЕПОДДЕРЖИВАЕМЫЙ ТИП ФАЙЛА"):
                            all_texts.append(inner_text)

                if all_texts:
                    return "\n\n--- [Content from " + file_name + "] ---\n\n".join(all_texts)
                else:
                    return f"Архив '{file_name}' не содержит поддерживаемых для анализа файлов."
            except rarfile.Error as e:
                logger.error(f"Could not process RAR file {file_name}: {e}")
                return f"Ошибка обработки RAR архива: {file_name}"

        elif mime_type.startswith("text/"):
            text = file_content.decode('utf-8')
            logger.info(f"Successfully decoded text file with {len(text)} characters.")
            return text

        else:
            logger.warning(f"Unsupported MIME type for text extraction: {mime_type}")
            return f"[Unsupported file format: {mime_type}]"

    except Exception as e:
        logger.error(f"Failed to extract text from file '{file_name}' with MIME type {mime_type}. Error: {e}", exc_info=True)
        return f"[Error processing file: {e}]"
    
    return ""
