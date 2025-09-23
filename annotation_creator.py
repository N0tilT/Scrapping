import os
import csv
from pathlib import Path
from typing import List, Tuple


def create_annotation_dataset(base_dir: str, output_file: str) -> str:
    """
    Создает файл-аннотацию для датасета с ревью фильмов.
    
    Args:
        base_dir: Путь к корневой директории датасета
        output_file: Путь для сохранения CSV-файла аннотации
        
    Returns:
        Сообщение о результате выполнения
    """
    try:
        base_path = Path(base_dir)
        classes = ["good", "bad"]
        
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Absolute Path", "Relative Path", "Class"])
            
            for class_name in classes:
                class_dir = base_path / class_name
                if not class_dir.exists():
                    return f"Ошибка: Директория {class_dir} не найдена"
                
                for file_path in class_dir.glob("*.txt"):
                    abs_path = file_path.resolve()
                    rel_path = file_path.relative_to(Path.cwd())
                    writer.writerow([abs_path, rel_path, class_name])
        
        return f"Аннотация успешно создана: {output_file}"
    
    except Exception as e:
        return f"Ошибка при создании аннотации: {str(e)}"


def get_dataset_stats(base_dir: str) -> Tuple[int, int]:
    """
    Получает статистику по датасету.
    
    Args:
        base_dir: Путь к корневой директории датасета
        
    Returns:
        Кортеж (количество good отзывов, количество bad отзывов)
    """
    base_path = Path(base_dir)
    good_count = len(list((base_path / "good").glob("*.txt")))
    bad_count = len(list((base_path / "bad").glob("*.txt")))
    return good_count, bad_count


if __name__ == "__main__":
    result = create_annotation_dataset("dataset", "annotation.csv")
    print(result)
    good, bad = get_dataset_stats("dataset")
    print(f"Good reviews: {good}, Bad reviews: {bad}")