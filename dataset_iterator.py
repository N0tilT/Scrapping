import pandas as pd
from pathlib import Path
from typing import Optional, Iterator


class ReviewIterator:
    """
    Итератор для последовательного доступа к отзывам определенного класса.
    """
    
    def __init__(self, annotation_file: str, class_label: str):
        """
        Инициализирует итератор.
        
        Args:
            annotation_file: Путь к файлу аннотации
            class_label: Метка класса ("good" или "bad")
        """
        self.annotation_file = annotation_file
        self.class_label = class_label
        self.df = pd.read_csv(annotation_file)
        self.instances = self.df[self.df["Class"] == class_label]["Absolute Path"].tolist()
        self.current_index = 0

    def get_next_instance(self) -> Optional[str]:
        """
        Возвращает следующий экземпляр указанного класса.
        
        Returns:
            Путь к следующему файлу или None, если экземпляры закончились
        """
        if self.current_index >= len(self.instances):
            return None
        
        next_path = self.instances[self.current_index]
        self.current_index += 1
        return next_path

    def reset_iterator(self) -> None:
        """Сбрасывает итератор в начальное состояние."""
        self.current_index = 0

    def get_remaining_count(self) -> int:
        """
        Возвращает количество оставшихся экземпляров.
        
        Returns:
            Количество оставшихся файлов
        """
        return len(self.instances) - self.current_index


class ReviewIteratorClass:
    """
    Класс-итератор для отзывов с поддержкой протокола итерации.
    """
    
    def __init__(self, annotation_file: str, class_label: str):
        self.annotation_file = annotation_file
        self.class_label = class_label
        self.df = pd.read_csv(annotation_file)
        self.instances = self.df[self.df["Class"] == class_label]["Absolute Path"].tolist()
        self.index = 0

    def __iter__(self) -> Iterator[str]:
        self.index = 0
        return self

    def __next__(self) -> str:
        if self.index >= len(self.instances):
            raise StopIteration
        next_path = self.instances[self.index]
        self.index += 1
        return next_path


def create_iterator(annotation_file: str, class_label: str) -> ReviewIterator:
    """
    Создает итератор для указанного класса.
    
    Args:
        annotation_file: Путь к файлу аннотации
        class_label: Метка класса ("good" или "bad")
        
    Returns:
        Объект итератора ReviewIterator
    """
    return ReviewIterator(annotation_file, class_label)


if __name__ == "__main__":
    iterator = create_iterator("annotation.csv", "good")
    print("Testing iterator:")
    for i in range(3):
        instance = iterator.get_next_instance()
        print(f"Instance {i+1}: {instance}")
    
    print("\nTesting iterator class:")
    for instance in ReviewIteratorClass("annotation.csv", "bad"):
        print(f"Instance: {instance}")