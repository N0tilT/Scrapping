import pytest
import tempfile
import os
import shutil


@pytest.fixture
def temp_directory():
    """Фикстура для создания временной директории"""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def sample_dataset_structure(temp_directory):
    """Фикстура для создания структуры тестового датасета"""
    dataset_dir = os.path.join(temp_directory, 'dataset')
    good_dir = os.path.join(dataset_dir, 'good')
    bad_dir = os.path.join(dataset_dir, 'bad')
    
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)
    
    for i in range(2):
        with open(os.path.join(good_dir, f'good_{i}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Фильм {i} (202{i})\nОтличный фильм с интересным сюжетом.")
        
        with open(os.path.join(bad_dir, f'bad_{i}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Фильм {i} (202{i})\nСкучный фильм без оригинального сюжета.")
    
    return {
        'dataset_dir': dataset_dir,
        'good_dir': good_dir,
        'bad_dir': bad_dir,
        'temp_dir': temp_directory
    }