import unittest
import tempfile
import os
import shutil
import csv
from pathlib import Path
import sys

sys.path.append('.')

from random_dataset_creator import create_random_dataset, get_random_stats


class TestRandomDatasetCreator(unittest.TestCase):
    
    def setUp(self):
        """Настройка тестовой среды"""
        self.test_dir = tempfile.mkdtemp()
        
        self.source_dir = os.path.join(self.test_dir, 'source_dataset')
        self.good_dir = os.path.join(self.source_dir, 'good')
        self.bad_dir = os.path.join(self.source_dir, 'bad')
        os.makedirs(self.good_dir, exist_ok=True)
        os.makedirs(self.bad_dir, exist_ok=True)
        
        self.create_test_files()
        
        self.target_dir = os.path.join(self.test_dir, 'random_dataset')
        self.annotation_file = os.path.join(self.test_dir, 'annotation_random.csv')
    
    def create_test_files(self):
        """Создание тестовых файлов"""
        for i in range(5):
            with open(os.path.join(self.good_dir, f'good_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Good review content {i}")
            
            with open(os.path.join(self.bad_dir, f'bad_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Bad review content {i}")
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.test_dir)
    
    def test_create_random_dataset_success(self):
        """Тест успешного создания случайного датасета"""
        result = create_random_dataset(self.source_dir, self.target_dir, self.annotation_file)
        
        self.assertIn("случайный датасет создан", result.lower())
        self.assertIn("аннотация создана", result.lower())
        
        self.assertTrue(os.path.exists(self.target_dir))
        
        files = list(Path(self.target_dir).glob('*.txt'))
        self.assertEqual(len(files), 10) 
        
        for file_path in files:
            filename = file_path.stem
            self.assertTrue(filename.isdigit())
            self.assertEqual(len(filename), 5)
    
    def test_create_random_dataset_annotation_content(self):
        """Тест содержимого файла аннотации"""
        create_random_dataset(self.source_dir, self.target_dir, self.annotation_file)
        
        self.assertTrue(os.path.exists(self.annotation_file))
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            self.assertEqual(len(rows), 10)
            
            classes = [row['Class'] for row in rows]
            self.assertEqual(classes.count('good'), 5)
            self.assertEqual(classes.count('bad'), 5)
            
            file_numbers = []
            for row in rows:
                abs_path = Path(row['Absolute Path'])
                file_number = abs_path.stem
                file_numbers.append(file_number)
            
            self.assertEqual(len(file_numbers), len(set(file_numbers)))
    
    def test_get_random_stats(self):
        """Тест получения статистики случайного датасета"""
        create_random_dataset(self.source_dir, self.target_dir, self.annotation_file)
        
        total_count = get_random_stats(self.target_dir)
        
        self.assertEqual(total_count, 10)
    
    def test_create_random_dataset_missing_source(self):
        """Тест с отсутствующей исходной директорией"""
        missing_dir = os.path.join(self.test_dir, 'nonexistent')
        result = create_random_dataset(missing_dir, self.target_dir, self.annotation_file)
        
        self.assertIn("ошибка", result.lower())
        self.assertIn("не найдена", result.lower())
    
    def test_file_content_preservation(self):
        """Тест сохранения содержимого файлов"""
        create_random_dataset(self.source_dir, self.target_dir, self.annotation_file)
        
        files = list(Path(self.target_dir).glob('*.txt'))
        self.assertGreater(len(files), 0)
        
        test_file = files[0]
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertTrue("review content" in content)
    
    def test_unique_file_numbers(self):
        """Тест уникальности номеров файлов"""
        create_random_dataset(self.source_dir, self.target_dir, self.annotation_file)
        
        files = list(Path(self.target_dir).glob('*.txt'))
        file_numbers = [f.stem for f in files]
        
        self.assertEqual(len(file_numbers), len(set(file_numbers)))
        
        for number_str in file_numbers:
            number = int(number_str)
            self.assertTrue(0 <= number <= 10000)


if __name__ == '__main__':
    unittest.main()