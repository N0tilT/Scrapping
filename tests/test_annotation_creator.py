import unittest
import tempfile
import os
import shutil
import csv
from pathlib import Path
import sys

sys.path.append('.')

from annotation_creator import create_annotation_dataset, get_dataset_stats


class TestAnnotationCreator(unittest.TestCase):
    
    def setUp(self):
        """Настройка тестовой среды"""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.test_dir, 'test_dataset')
        
        self.good_dir = os.path.join(self.dataset_dir, 'good')
        self.bad_dir = os.path.join(self.dataset_dir, 'bad')
        os.makedirs(self.good_dir, exist_ok=True)
        os.makedirs(self.bad_dir, exist_ok=True)
        
        self.create_test_files()
        
        self.annotation_file = os.path.join(self.test_dir, 'annotation.csv')
    
    def create_test_files(self):
        """Создание тестовых файлов отзывов"""
        for i in range(3):
            with open(os.path.join(self.good_dir, f'good_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Хороший фильм {i}\nОтличный сюжет и актеры.")
            
            with open(os.path.join(self.bad_dir, f'bad_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Плохой фильм {i}\nСлабый сценарий.")
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.test_dir)
    
    def test_create_annotation_dataset_success(self):
        """Тест успешного создания аннотации"""
        result = create_annotation_dataset(self.dataset_dir, self.annotation_file)
        
        self.assertIn("успешно создана", result.lower())
        
        self.assertTrue(os.path.exists(self.annotation_file))
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            self.assertEqual(rows[0], ['Absolute Path', 'Relative Path', 'Class'])
            
            self.assertEqual(len(rows) - 1, 6)
            
            classes = [row[2] for row in rows[1:]]
            self.assertIn('good', classes)
            self.assertIn('bad', classes)
    
    def test_create_annotation_dataset_missing_directory(self):
        """Тест с отсутствующей директорией"""
        missing_dir = os.path.join(self.test_dir, 'nonexistent')
        result = create_annotation_dataset(missing_dir, self.annotation_file)
        
        self.assertIn("ошибка", result.lower())
        self.assertIn("не найдена", result.lower())
    
    def test_get_dataset_stats(self):
        """Тест получения статистики датасета"""
        good_count, bad_count = get_dataset_stats(self.dataset_dir)
        
        self.assertEqual(good_count, 3)
        self.assertEqual(bad_count, 3)
    
    def test_get_dataset_stats_empty(self):
        """Тест статистики пустого датасета"""
        empty_dir = os.path.join(self.test_dir, 'empty_dataset')
        os.makedirs(os.path.join(empty_dir, 'good'))
        os.makedirs(os.path.join(empty_dir, 'bad'))
        
        good_count, bad_count = get_dataset_stats(empty_dir)
        
        self.assertEqual(good_count, 0)
        self.assertEqual(bad_count, 0)
    
    def test_annotation_file_content(self):
        """Тест содержимого файла аннотации"""
        create_annotation_dataset(self.dataset_dir, self.annotation_file)
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                self.assertIsNotNone(row['Absolute Path'])
                self.assertIsNotNone(row['Relative Path'])
                self.assertIn(row['Class'], ['good', 'bad'])
                
                self.assertTrue(os.path.exists(row['Absolute Path']))


if __name__ == '__main__':
    unittest.main()