import unittest
import pandas as pd
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
import sys
import numpy as np

sys.path.append('.')

from analytics import TextAnalyzer


class TestTextAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Настройка тестовой среды"""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, 'output')
        
        self.good_dir = os.path.join(self.test_dir, 'dataset', 'good')
        self.bad_dir = os.path.join(self.test_dir, 'dataset', 'bad')
        os.makedirs(self.good_dir, exist_ok=True)
        os.makedirs(self.bad_dir, exist_ok=True)
        
        self.create_test_reviews()
        
        self.analyzer = TextAnalyzer(output_dir=self.output_dir, threshold=10)
    
    def create_test_reviews(self):
        """Создание тестовых отзывов"""
        good_reviews = [
            "Отличный фильм (2020)\nЭто прекрасный фильм с интересным сюжетом и великолепной актерской игрой.",
            "Великолепное кино (2019)\nПотрясающая картина, которая заставляет задуматься о важных вещах.",
            "Шедевр (2021)\nНевероятно талантливая режиссура и глубокий смысл."
        ]
        
        # Отрицательные отзывы  
        bad_reviews = [
            "Ужасный фильм (2020)\nСкучный и предсказуемый сюжет, плохая актерская игра.",
            "Разочарование (2019)\nОжидал большего, фильм не оправдал ожидания.",
            "Слабая картина (2021)\nПлохой сценарий и слабая режиссура."
        ]
        
        for i, review in enumerate(good_reviews):
            with open(os.path.join(self.good_dir, f'good_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(review)
                
        for i, review in enumerate(bad_reviews):
            with open(os.path.join(self.bad_dir, f'bad_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(review)
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.test_dir)
    
    def test_parse_movie_info(self):
        """Тест парсинга информации о фильме"""
        test_cases = [
            ("Фильм (2020)\nТекст отзыва", ("Фильм", 2020)),
            ("Фильм, 2019\nТекст отзыва", ("Фильм", 2019)),
            ("Просто текст без года", ("Неизвестный фильм", None))
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.analyzer._parse_movie_info(text)
                self.assertEqual(result, expected)
    
    def test_load_reviews_from_folders(self):
        """Тест загрузки отзывов из папок"""
        df = self.analyzer.load_reviews_from_folders(self.good_dir, self.bad_dir)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('review_type', df.columns)
        self.assertIn('text', df.columns)
        self.assertIn('movie_title', df.columns)
        self.assertIn('year', df.columns)
        
        review_types = df['review_type'].unique()
        self.assertTrue(any(t in review_types for t in ['good', 'bad']))
    
    def test_preprocess_dataframe(self):
        """Тест предобработки DataFrame"""
        test_data = {
            'review_type': ['good', 'bad', 'good'],
            'text': ['Отличный фильм', 'Плохой фильм', ''],
            'movie_title': ['Фильм 1', 'Фильм 2', 'Фильм 3'],
            'year': [2020, 2019, 2021]
        }
        df = pd.DataFrame(test_data)
        
        processed_df = self.analyzer.preprocess_dataframe(df)
        
        self.assertEqual(len(processed_df), 2)
        self.assertIn('word_count', processed_df.columns)
        self.assertIn('class_label', processed_df.columns)
        self.assertEqual(processed_df['class_label'].iloc[0], 1)
        self.assertEqual(processed_df['class_label'].iloc[1], 0)
    
    def test_count_words(self):
        """Тест подсчета слов"""
        text = "Это тестовый текст с несколькими словами."
        word_count = self.analyzer._count_words(text)
        self.assertIsInstance(word_count, int)
        self.assertGreater(word_count, 0)
    
    @patch('matplotlib.pyplot.show')
    def test_create_word_histogram(self, mock_show):
        """Тест создания гистограммы слов"""
        test_data = {
            'review_type': ['good', 'bad', 'good', 'bad'],
            'text': [
                'отличный прекрасный великолепный фильм',
                'плохой ужасный скучный фильм', 
                'прекрасный замечательный шедевр',
                'скучный разочарование плохой'
            ],
            'movie_title': ['Ф1', 'Ф2', 'Ф3', 'Ф4'],
            'year': [2020, 2020, 2021, 2021],
            'class_label': [1, 0, 1, 0]
        }
        df = pd.DataFrame(test_data)
        
        words, counts = self.analyzer.create_word_histogram(df, class_label=1, top_n=5)
        
        if words and counts:
            self.assertIsInstance(words, tuple)
            self.assertIsInstance(counts, tuple)
            self.assertEqual(len(words), len(counts))
    
    def test_filter_functions(self):
        """Тест функций фильтрации"""
        test_data = {
            'review_type': ['good', 'bad', 'good', 'bad'],
            'text': ['Текст 1', 'Текст 2', 'Текст 3', 'Текст 4'],
            'movie_title': ['Фильм 1', 'Фильм 2', 'Фильм 1', 'Фильм 3'],
            'year': [2020, 2020, 2021, 2021],
            'word_count': [50, 150, 75, 200],
            'class_label': [1, 0, 1, 0]
        }
        df = pd.DataFrame(test_data)
        
        filtered = self.analyzer.filter_by_word_count(df, max_words=100)
        self.assertTrue(all(filtered['word_count'] <= 100))
        
        positive = self.analyzer.filter_by_class_label(df, class_label=1)
        self.assertTrue(all(positive['class_label'] == 1))
        
        year_2020 = self.analyzer.filter_by_year(df, year=2020)
        self.assertTrue(all(year_2020['year'] == 2020))
        
        movie_1 = self.analyzer.filter_by_movie(df, movie_title='Фильм 1')
        self.assertTrue(all(movie_1['movie_title'] == 'Фильм 1'))
    
    def test_statistics_functions(self):
        """Тест функций статистики"""
        test_data = {
            'review_type': ['good', 'bad', 'good', 'bad'],
            'text': ['Текст 1', 'Текст 2', 'Текст 3', 'Текст 4'],
            'movie_title': ['Фильм 1', 'Фильм 2', 'Фильм 1', 'Фильм 3'],
            'year': [2020, 2020, 2021, 2021],
            'word_count': [50, 150, 75, 200],
            'class_label': [1, 0, 1, 0]
        }
        df = pd.DataFrame(test_data)
        
        group_stats = self.analyzer.calculate_group_statistics(df)
        self.assertIsInstance(group_stats, pd.DataFrame)
        
        year_stats = self.analyzer.calculate_year_statistics(df)
        self.assertIsInstance(year_stats, pd.DataFrame)
        
        movie_stats = self.analyzer.calculate_movie_statistics(df)
        self.assertIsInstance(movie_stats, pd.DataFrame)
    
    def test_preprocess_text(self):
        """Тест предобработки текста"""
        text = "Это тестовый текст с некоторыми словами для проверки работы функции."
        lemmas = self.analyzer.preprocess_text(text)
        
        self.assertIsInstance(lemmas, list)
        for lemma in lemmas:
            self.assertNotIn(lemma, self.analyzer.stop_words)
            self.assertGreater(len(lemma), 2)
    
    @patch('matplotlib.pyplot.savefig')
    def test_analyze_texts_comprehensive(self, mock_savefig):
        """Тест комплексного анализа"""
        df = self.analyzer.analyze_texts_comprehensive(self.good_dir, self.bad_dir)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('word_count', df.columns)
        self.assertIn('class_label', df.columns)


class TestTextAnalyzerEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.analyzer = TextAnalyzer()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_empty_dataset(self):
        """Тест с пустым датасетом"""
        empty_dir = os.path.join(self.test_dir, 'empty')
        os.makedirs(empty_dir)
        
        df = self.analyzer.load_reviews_from_folders(empty_dir, empty_dir)
        self.assertEqual(len(df), 0)
        
        processed_df = self.analyzer.preprocess_dataframe(df)
        self.assertEqual(len(processed_df), 0)
    
    def test_invalid_paths(self):
        """Тест с несуществующими путями"""
        invalid_path = os.path.join(self.test_dir, 'nonexistent')
        df = self.analyzer.load_reviews_from_folders(invalid_path, invalid_path)
        self.assertEqual(len(df), 0)
    
    def test_malformed_reviews(self):
        """Тест с некорректно оформленными отзывами"""
        malformed_dir = os.path.join(self.test_dir, 'malformed')
        os.makedirs(malformed_dir)
        
        with open(os.path.join(malformed_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write("Просто текст без года\nИ еще какой-то текст")
        
        df = self.analyzer.load_reviews_from_folders(malformed_dir, malformed_dir)
        self.assertGreater(len(df), 0)
        self.assertEqual(df.iloc[0]['movie_title'], "Неизвестный фильм")


if __name__ == '__main__':
    unittest.main()