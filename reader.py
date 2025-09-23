def read_file_content(file_path: str) -> str:
    """
    Читает содержимое текстового файла.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Содержимое файла или сообщение об ошибке
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content if content.strip() else "Файл пуст"
    except Exception as e:
        return f"Ошибка чтения файла: {str(e)}"