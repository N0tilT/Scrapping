"""
Скрипт для запуска всех unit-тестов
"""

import unittest
import sys
import os

def run_all_tests():
    """Запуск всех тестов"""
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())