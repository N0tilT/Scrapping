import unittest
import tempfile
import os
import shutil
import csv
from pathlib import Path
import sys

sys.path.append('.')

from dataset_reorganizer import reorganize_dataset, get_reorganized_stats


class TestDatasetReorganizer(unittest.TestCase):
    
    def setUp(self):
        """Настройка тестовой среды"""
        self.test_dir = tempfile.mkdtemp()
        
        self.source_dir = os.path.join(self.test_dir, 'source_dataset')
        self.good_dir = os.path.join(self.source_dir, 'good')
        self.bad_dir = os.path.join(self.source_dir, 'bad')
        os.makedirs(self.good_dir, exist_ok=True)
        os.makedirs(self.bad_dir, exist_ok=True)
        
        self.create_test_files()
        
        self.target_dir = os.path.join(self.test_dir, 'target_dataset')
        self.annotation_file = os.path.join(self.test_dir, 'annotation.csv')
    
    def create_test_files(self):
        """Создание тестовых файлов"""
        for i in range(3):
            with open(os.path.join(self.good_dir, f'review_good_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Good review content {i}")
            
            with open(os.path.join(self.bad_dir, f'review_bad_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Bad review content {i}")
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.test_dir)
    
    def test_reorganize_dataset_success(self):
        """Тест успешной реорганизации датасета"""
        result = reorganize_dataset(self.source_dir, self.target_dir, self.annotation_file)
        
        self.assertIn("реорганизован", result.lower())
        self.assertIn("аннотация создана", result.lower())
        
        self.assertTrue(os.path.exists(self.target_dir))
        
        good_files = list(Path(self.target_dir).glob('good_*.txt'))
        bad_files = list(Path(self.target_dir).glob('bad_*.txt'))
        
        self.assertEqual(len(good_files), 3)
        self.assertEqual(len(bad_files), 3)
        
        for i, file_path in enumerate(sorted(good_files)):
            expected_name = f"good_{i:04d}.txt"
            self.assertEqual(file_path.name, expected_name)
        
        for i, file_path in enumerate(sorted(bad_files)):
            expected_name = f"bad_{i:04d}.txt"
            self.assertEqual(file_path.name, expected_name)
    
    def test_reorganize_dataset_annotation_content(self):
        """Тест содержимого файла аннотации после реорганизации"""
        reorganize_dataset(self.source_dir, self.target_dir, self.annotation_file)
        
        self.assertTrue(os.path.exists(self.annotation_file))
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            self.assertEqual(len(rows), 6)
            
            classes = [row['Class'] for row in rows]
            self.assertEqual(classes.count('good'), 3)
            self.assertEqual(classes.count('bad'), 3)
            
            for row in rows:
                self.assertTrue(os.path.exists(row['Absolute Path']))
    
    def test_get_reorganized_stats(self):
        """Тест получения статистики реорганизованного датасета"""
        reorganize_dataset(self.source_dir, self.target_dir, self.annotation_file)
        
        good_count, bad_count = get_reorganized_stats(self.target_dir)
        
        self.assertEqual(good_count, 3)
        self.assertEqual(bad_count, 3)
    
    def test_reorganize_dataset_missing_source(self):
        """Тест с отсутствующей исходной директорией"""
        missing_dir = os.path.join(self.test_dir, 'nonexistent')
        result = reorganize_dataset(missing_dir, self.target_dir, self.annotation_file)
        
        self.assertIn("ошибка", result.lower())
        self.assertIn("не найдена", result.lower())
    
    def test_reorganize_dataset_empty_directories(self):
        """Тест с пустыми директориями"""
        empty_dir = os.path.join(self.test_dir, 'empty_dataset')
        os.makedirs(os.path.join(empty_dir, 'good'))
        os.makedirs(os.path.join(empty_dir, 'bad'))
        
        result = reorganize_dataset(empty_dir, self.target_dir, self.annotation_file)
        
        self.assertIn("реорганизован", result.lower())
        
        files = list(Path(self.target_dir).glob('*.txt'))
        self.assertEqual(len(files), 0)
    
    def test_file_content_preservation(self):
        """Тест сохранения содержимого файлов при реорганизации"""
        reorganize_dataset(self.source_dir, self.target_dir, self.annotation_file)
        
        test_file = os.path.join(self.target_dir, 'good_0000.txt')
        self.assertTrue(os.path.exists(test_file))
        
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("Good review content", content)


if __name__ == '__main__':
    unittest.main()