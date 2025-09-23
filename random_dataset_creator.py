import random
import shutil
import csv
from pathlib import Path
from typing import Tuple, Set


def create_random_dataset(source_dir: str, target_dir: str, annotation_file: str) -> str:
    """
    Создает датасет со случайными номерами файлов.
    
    Args:
        source_dir: Исходная директория датасета
        target_dir: Целевая директория для рандомизированного датасета
        annotation_file: Путь для сохранения файла аннотации
        
    Returns:
        Сообщение о результате выполнения
    """
    try:
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        used_numbers: Set[int] = set()
        
        with open(annotation_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Absolute Path", "Relative Path", "Class"])
            
            for class_name in ["good", "bad"]:
                class_dir = source_path / class_name
                if not class_dir.exists():
                    return f"Ошибка: Директория {class_dir} не найдена"
                
                files = list(class_dir.glob("*.txt"))
                for file_path in files:
                    while True:
                        num = random.randint(0, 10000)
                        if num not in used_numbers:
                            used_numbers.add(num)
                            break
                    
                    new_name = f"{num:05d}.txt"
                    new_path = target_path / new_name
                    shutil.copy2(file_path, new_path)
                    
                    abs_path = new_path.resolve()
                    rel_path = new_path.relative_to(Path.cwd())
                    writer.writerow([abs_path, rel_path, class_name])
        
        return f"Случайный датасет создан: {target_dir}"
    
    except Exception as e:
        return f"Ошибка при создании случайного датасета: {str(e)}"


def get_random_stats(target_dir: str) -> int:
    """
    Получает статистику по случайному датасету.
    
    Args:
        target_dir: Путь к случайному датасету
        
    Returns:
        Общее количество файлов
    """
    target_path = Path(target_dir)
    return len(list(target_path.glob("*.txt")))


if __name__ == "__main__":
    result = create_random_dataset("dataset", "dataset_random", "annotation_random.csv")
    print(result)
    count = get_random_stats("dataset_random")
    print(f"Total files: {count}")