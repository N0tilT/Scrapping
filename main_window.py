import flet as ft
from reader import read_file_content
from pathlib import Path
from annotation_creator import create_annotation_dataset, get_dataset_stats
from dataset_reorganizer import reorganize_dataset, get_reorganized_stats
from random_dataset_creator import create_random_dataset, get_random_stats
from dataset_iterator import ReviewIterator

class MovieReviewApp:
    """Основной класс приложения для работы с датасетом ревью фильмов."""
    
    def __init__(self):
        self.current_iterator = None
        self.annotation_file_path = ""
        
    def main(self, page: ft.Page):
        """Основная функция инициализации интерфейса."""
        page.title = "Movie Review Dataset Manager"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 20
        page.scroll = ft.ScrollMode.AUTO
        
        self.source_dir_field = ft.TextField(
            label="Путь к исходному датасету",
            value="dataset",
            width=400
        )
        
        self.annotation_file_field = ft.TextField(
            label="Путь для файла аннотации",
            value="annotation.csv",
            width=400
        )
        
        self.target_dir_field = ft.TextField(
            label="Путь к целевой директории",
            value="dataset_reorganized",
            width=400
        )
        
        self.random_dir_field = ft.TextField(
            label="Путь для случайного датасету",
            value="dataset_random",
            width=400
        )
        
        self.result_text = ft.Text("", size=16, color=ft.Colors.BLUE)
        self.stats_text = ft.Text("", size=14, color=ft.Colors.GREEN)
        
        create_annotation_btn = ft.ElevatedButton(
            "Создать аннотацию",
            on_click=self.create_annotation,
            icon=ft.Icons.CREATE
        )
        
        reorganize_btn = ft.ElevatedButton(
            "Реорганизовать датасет",
            on_click=self.reorganize_dataset,
            icon=ft.Icons.REORDER
        )
        
        random_dataset_btn = ft.ElevatedButton(
            "Создать случайный датасет",
            on_click=self.create_random_dataset,
            icon=ft.Icons.CASINO
        )
        
        self.iterator_class_field = ft.TextField(
            label="Класс для итератора (good/bad)",
            value="good",
            width=200
        )
        
        self.iterator_file_field = ft.TextField(
            label="Файл аннотации для итератора",
            value="annotation.csv",
            width=400
        )
        
        self.iterator_result = ft.Text("", size=14, color=ft.Colors.PURPLE)
        
        self.file_content_header = ft.Text("Содержимое файла:", 
                                          size=16, 
                                          weight=ft.FontWeight.BOLD,
                                          visible=False)
        
        self.file_name_text = ft.Text("", size=14, color=ft.Colors.BROWN, visible=False)
        
        self.file_content_text = ft.Text("", size=12, selectable=True)
        self.file_content_area = ft.Column(
            controls=[self.file_content_text],
            scroll=ft.ScrollMode.ALWAYS,
            height=200,
            width=600
        )
        
        self.file_content_container = ft.Container(
            content=self.file_content_area,
            padding=10,
            border=ft.border.all(1, ft.Colors.GREY_400),
            border_radius=5,
            width=600,
            height=200,
            visible=False
        )
        
        next_instance_btn = ft.ElevatedButton(
            "Следующий экземпляр",
            on_click=self.next_instance,
            icon=ft.Icons.NEXT_PLAN
        )
        
        reset_iterator_btn = ft.ElevatedButton(
            "Сбросить итератор",
            on_click=self.reset_iterator,
            icon=ft.Icons.REFRESH
        )
        
        page.add(
            ft.Text("Менеджер датасета ревью фильмов", size=24, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            
            ft.Text("Основные операции:", size=18, weight=ft.FontWeight.BOLD),
            self.source_dir_field,
            
            ft.Row([
                self.annotation_file_field,
                create_annotation_btn
            ]),
            
            ft.Row([
                self.target_dir_field,
                reorganize_btn
            ]),
            
            ft.Row([
                self.random_dir_field,
                random_dataset_btn
            ]),
            
            self.result_text,
            self.stats_text,
            ft.Divider(),
            
            ft.Text("Итератор по классам:", size=18, weight=ft.FontWeight.BOLD),
            ft.Row([
                self.iterator_class_field,
                self.iterator_file_field
            ]),
            
            ft.Row([
                next_instance_btn,
                reset_iterator_btn
            ]),
            
            self.iterator_result,
            self.file_content_header,
            self.file_name_text,
            self.file_content_container
        )
    
    def create_annotation(self, e):
        """Обработчик создания аннотации."""
        source_dir = self.source_dir_field.value
        annotation_file = self.annotation_file_field.value
        
        if not source_dir or not annotation_file:
            self.show_result("Ошибка: Заполните все поля")
            return
        
        result = create_annotation_dataset(source_dir, annotation_file)
        self.show_result(result)
        
        good, bad = get_dataset_stats(source_dir)
        self.stats_text.value = f"Статистика: Good - {good}, Bad - {bad}"
        self.stats_text.update()
        
        self.annotation_file_path = annotation_file
    
    def reorganize_dataset(self, e):
        """Обработчик реорганизации датасета."""
        source_dir = self.source_dir_field.value
        target_dir = self.target_dir_field.value
        annotation_file = f"annotation_{Path(target_dir).name}.csv"
        
        result = reorganize_dataset(source_dir, target_dir, annotation_file)
        self.show_result(result)
        
        good, bad = get_reorganized_stats(target_dir)
        self.stats_text.value = f"Реорганизовано: Good - {good}, Bad - {bad}"
        self.stats_text.update()
    
    def create_random_dataset(self, e):
        """Обработчик создания случайного датасета."""
        source_dir = self.source_dir_field.value
        target_dir = self.random_dir_field.value
        annotation_file = f"annotation_{Path(target_dir).name}.csv"
        
        result = create_random_dataset(source_dir, target_dir, annotation_file)
        self.show_result(result)
        
        count = get_random_stats(target_dir)
        self.stats_text.value = f"Создано файлов: {count}"
        self.stats_text.update()
    
    def next_instance(self, e):
        """Обработчик получения следующего экземпляра через итератор."""
        class_label = self.iterator_class_field.value
        annotation_file = self.iterator_file_field.value
        
        if not class_label or not annotation_file:
            self.iterator_result.value = "Ошибка: Заполните поля класса и файла аннотации"
            self.iterator_result.update()
            return
        
        if self.current_iterator is None or self.annotation_file_path != annotation_file:
            self.current_iterator = ReviewIterator(annotation_file, class_label)
            self.annotation_file_path = annotation_file
        
        instance = self.current_iterator.get_next_instance()
        
        if instance is None:
            self.iterator_result.value = "Экземпляры закончились"
            self.file_content_header.visible = False
            self.file_name_text.visible = False
            self.file_content_container.visible = False
        else:
            remaining = self.current_iterator.get_remaining_count()
            self.iterator_result.value = f"Следующий: {Path(instance).name}\nОсталось: {remaining}"
            
            content = read_file_content(instance)
            file_name = Path(instance).name
            
            self.file_name_text.value = f"Файл: {file_name}"
            self.file_content_text.value = content
            
            self.file_content_header.visible = True
            self.file_name_text.visible = True
            self.file_content_container.visible = True
        
        self.iterator_result.update()
        self.file_content_header.update()
        self.file_name_text.update()
        self.file_content_container.update()
    
    def reset_iterator(self, e):
        """Обработчик сброса итератора."""
        if self.current_iterator:
            self.current_iterator.reset_iterator()
            self.iterator_result.value = "Итератор сброшен"
            self.iterator_result.update()
            
            self.file_content_header.visible = False
            self.file_name_text.visible = False
            self.file_content_container.visible = False
            self.file_content_header.update()
            self.file_name_text.update()
            self.file_content_container.update()
    
    def show_result(self, message: str):
        """Отображает результат операции."""
        self.result_text.value = message
        self.result_text.update()


def main():
    """Запуск приложения."""
    app = MovieReviewApp()
    ft.app(target=app.main)


if __name__ == "__main__":
    main()