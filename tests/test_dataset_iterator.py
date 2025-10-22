import unittest
import tempfile
import os
import shutil
import csv
import pandas as pd
import sys

sys.path.append('.')

from dataset_iterator import ReviewIterator, ReviewIteratorClass, create_iterator


class TestDatasetIterator(unittest.TestCase):
    
    def setUp(self):
        """Настройка тестовой среды"""
        self.test_dir = tempfile.mkdtemp()
        self.annotation_file = os.path.join(self.test_dir, 'annotation.csv')
        
        self.create_test_annotation()
    
    def create_test_annotation(self):
        """Создание тестовой аннотации"""
        with open(self.annotation_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Absolute Path', 'Relative Path', 'Class'])
            
            for i in range(5):
                good_file = os.path.join(self.test_dir, f'good_{i}.txt')
                bad_file = os.path.join(self.test_dir, f'bad_{i}.txt')
            
                with open(good_file, 'w', encoding='utf-8') as gf:
                    gf.write(f"Good review {i}")
                with open(bad_file, 'w', encoding='utf-8') as bf:
                    bf.write(f"Bad review {i}")
                
                writer.writerow([good_file, f'good_{i}.txt', 'good'])
                writer.writerow([bad_file, f'bad_{i}.txt', 'bad'])
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.test_dir)
    
    def test_review_iterator_initialization(self):
        """Тест инициализации итератора"""
        iterator = ReviewIterator(self.annotation_file, 'good')
        
        self.assertEqual(iterator.annotation_file, self.annotation_file)
        self.assertEqual(iterator.class_label, 'good')
        self.assertEqual(iterator.current_index, 0)
        self.assertEqual(len(iterator.instances), 5)
    
    def test_review_iterator_get_next_instance(self):
        """Тест получения следующего экземпляра"""
        iterator = ReviewIterator(self.annotation_file, 'good')
        
        first_instance = iterator.get_next_instance()
        self.assertIsNotNone(first_instance)
        self.assertTrue(os.path.exists(first_instance))
        
        for i in range(4):
            instance = iterator.get_next_instance()
            self.assertIsNotNone(instance)
        
        null_instance = iterator.get_next_instance()
        self.assertIsNone(null_instance)
    
    def test_review_iterator_reset(self):
        """Тест сброса итератора"""
        iterator = ReviewIterator(self.annotation_file, 'bad')
        
        iterator.get_next_instance()
        iterator.get_next_instance()
        
        self.assertEqual(iterator.current_index, 2)
        
        iterator.reset_iterator()
        self.assertEqual(iterator.current_index, 0)
    
    def test_review_iterator_get_remaining_count(self):
        """Тест получения количества оставшихся экземпляров"""
        iterator = ReviewIterator(self.annotation_file, 'good')
        
        self.assertEqual(iterator.get_remaining_count(), 5)
        
        iterator.get_next_instance()
        self.assertEqual(iterator.get_remaining_count(), 4)
        
        for i in range(4):
            iterator.get_next_instance()
        
        self.assertEqual(iterator.get_remaining_count(), 0)
    
    def test_review_iterator_class_protocol(self):
        """Тест протокола итератора для ReviewIteratorClass"""
        iterator = ReviewIteratorClass(self.annotation_file, 'good')
        
        instances = list(iterator)
        self.assertEqual(len(instances), 5)
        
        for instance in instances:
            self.assertTrue(os.path.exists(instance))
    
    def test_create_iterator_function(self):
        """Тест функции создания итератора"""
        iterator = create_iterator(self.annotation_file, 'bad')
        
        self.assertIsInstance(iterator, ReviewIterator)
        self.assertEqual(iterator.class_label, 'bad')
    
    def test_iterator_with_invalid_class(self):
        """Тест итератора с несуществующим классом"""
        iterator = ReviewIterator(self.annotation_file, 'nonexistent')
        
        self.assertEqual(len(iterator.instances), 0)
        self.assertIsNone(iterator.get_next_instance())
        self.assertEqual(iterator.get_remaining_count(), 0)
    
    def test_iterator_with_invalid_annotation_file(self):
        """Тест с несуществующим файлом аннотации"""
        with self.assertRaises(FileNotFoundError):
            ReviewIterator('nonexistent.csv', 'good')


if __name__ == '__main__':
    unittest.main()