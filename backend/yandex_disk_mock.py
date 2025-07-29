def list_files(path: str) -> list[str]:
    """
    Simulates listing files in a directory.
    NOTE: The path argument is ignored in this mock version.
    """
    return ['Проект_Альфа_чертеж_1.pdf', 'Спецификация_Альфа.docx', 'Примечания.txt']


def get_file_content(file_path: str) -> str:
    """
    Simulates reading the content of a file based on its path.
    """
    if ".pdf" in file_path:
        return f"Mock content of a PDF file for {file_path}"
    elif ".docx" in file_path:
        return f"Mock content of a DOCX file for {file_path}"
    else:
        return f"Generic mock content for file: {file_path}"
