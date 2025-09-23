import shutil
import csv
from pathlib import Path
from typing import Tuple


def reorganize_dataset(source_dir: str, target_dir: str, annotation_file: str) -> str:
    """
    Реорганизует датасет, переименовывая файлы с включением имени класса.
    
    Args:
        source_dir: Исходная директория датасета
        target_dir: Целевая директория для реорганизованного датасета
        annotation_file: Путь для сохранения файла аннотации
        
    Returns:
        Сообщение о результате выполнения
    """
    try:
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        with open(annotation_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Absolute Path", "Relative Path", "Class"])
            
            for class_name in ["good", "bad"]:
                class_dir = source_path / class_name
                if not class_dir.exists():
                    return f"Ошибка: Директория {class_dir} не найдена"
                
                files = list(class_dir.glob("*.txt"))
                for i, file_path in enumerate(files):
                    new_name = f"{class_name}_{i:04d}.txt"
                    new_path = target_path / new_name
                    shutil.copy2(file_path, new_path)
                    
                    abs_path = new_path.resolve()
                    try:
                        rel_path = new_path.relative_to(Path.cwd())
                    except ValueError:
                        rel_path = abs_path
                    
                    writer.writerow([abs_path, rel_path, class_name])
        
        return f"Датасет реорганизован: {target_dir}. Аннотация создана: {annotation_file}"
    
    except Exception as e:
        return f"Ошибка при реорганизации датасета: {str(e)}"


def get_reorganized_stats(target_dir: str) -> Tuple[int, int]:
    """
    Получает статистику по реорганизованному датасету.
    
    Args:
        target_dir: Путь к реорганизованному датасету
        
    Returns:
        Кортеж (количество good отзывов, количество bad отзывов)
    """
    target_path = Path(target_dir)
    good_count = len(list(target_path.glob("good_*.txt")))
    bad_count = len(list(target_path.glob("bad_*.txt")))
    return good_count, bad_count


if __name__ == "__main__":
    result = reorganize_dataset("dataset", "dataset_reorganized", "annotation_new.csv")
    print(result)
    good, bad = get_reorganized_stats("dataset_reorganized")
    print(f"Good reviews: {good}, Bad reviews: {bad}")