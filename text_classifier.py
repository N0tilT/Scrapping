import os
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from analytics import TextAnalyzer

class TextClassifier:
    def __init__(self, output_dir='model_results', log_file='classification_log.txt'):
        self.analyzer = TextAnalyzer(threshold=1000)
        self.output_dir = output_dir
        self.log_file = log_file
        self.best_model = None
        self.vectorizer = None
        self.has_probabilities = False
        
        os.makedirs(output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, log_file)),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess_data(self, good_path='dataset/good', bad_path='dataset/bad'):
        """Загрузка и предобработка данных"""
        self.logger.info("Загрузка данных...")
        
        df = self.analyzer.load_reviews_from_folders(good_path, bad_path)
        df = self.analyzer.preprocess_dataframe(df)
        
        texts = df['text'].tolist()
        labels = df['class_label'].tolist()
        
        self.logger.info(f"Загружено {len(texts)} отзывов")
        self.logger.info(f"Распределение классов: {pd.Series(labels).value_counts().to_dict()}")
        
        return texts, labels
    
    def preprocess_texts(self, texts):
        """Предобработка текстов с использованием Natasha"""
        self.logger.info("Предобработка текстов...")
        processed_texts = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                self.logger.info(f"Обработано {i}/{len(texts)} текстов")
            
            lemmas = self.analyzer.preprocess_text(text)
            processed_text = ' '.join(lemmas)
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def split_dataset(self, texts, labels, test_size=0.2, val_size=0.15, random_state=42):
        """Разделение данных на train, validation и test"""
        self.logger.info("Разделение данных...")
        
        X_temp, X_demo, y_temp, y_demo = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        self.logger.info(f"Размер тренировочных данных: {len(X_train)}")
        self.logger.info(f"Размер валидационных данных: {len(X_val)}")
        self.logger.info(f"Размер демонстрационных данных: {len(X_demo)}")
        
        return X_train, X_val, X_demo, y_train, y_val, y_demo
    
    def create_model(self, model_type='logistic', **params):
        """Создание модели с TF-IDF векторизацией"""
        base_model = None
        
        if model_type == 'logistic':
            base_model = LogisticRegression(**params, random_state=42)
            self.has_probabilities = True
        elif model_type == 'svm':
            base_model = SVC(**params, random_state=42)
            if params.get('probability', False):
                self.has_probabilities = True
            else:
                self.has_probabilities = False
        elif model_type == 'random_forest':
            base_model = RandomForestClassifier(**params, random_state=42)
            self.has_probabilities = True
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words=list(self.analyzer.stop_words)
            )),
            ('classifier', base_model)
        ])
        
        return pipeline
    
    def train_model(self, model, X_train, y_train, X_val, y_val):
        """Обучение модели"""
        self.logger.info("Обучение модели...")
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        self.logger.info(f"Точность на тренировочных данных: {train_score:.4f}")
        self.logger.info(f"Точность на валидационных данных: {val_score:.4f}")
        
        return train_score, val_score
    
    def test_model(self, model, X_test, y_test, save_predictions=True,title=""):
        """Тестирование модели"""
        self.logger.info("Тестирование модели...")
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"Точность на тестовых данных: {accuracy:.4f}")
        self.logger.info("\nОтчет классификации:")
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        if save_predictions:
            results_df = pd.DataFrame({
                'text': X_test,
                'true_label': y_test,
                'predicted_label': y_pred,
                'correct': y_test == y_pred
            })
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.output_dir, f'demo_predictions_{timestamp}.txt')
            results_df.to_csv(results_file, index=False, sep='\t', encoding='utf-8')
            self.logger.info(f"Предсказания сохранены в: {results_file}")
        
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, accuracy,title)
        
        return accuracy, y_pred
    
    def plot_confusion_matrix(self, cm, accuracy,title=""):
        """Визуализация матрицы ошибок"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Матрица ошибок\nТочность: {accuracy:.4f}')
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.output_dir, f'{f"{title}_" if title!="" else ""}confusion_matrix_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Матрица ошибок сохранена в: {plot_file}")
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, X_demo, y_demo):
        """Настройка гиперпараметров"""
        self.logger.info("=== НАЧАЛО НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ ===")
        
        experiments = []
        
        model_params = [
            ('logistic', {'C': 1.0, 'max_iter': 10000}),
            ('svm', {'C': 1.0, 'kernel': 'linear', 'probability': True}), 
            ('svm', {'C': 1.0, 'kernel': 'linear', 'probability': False}),
            ('random_forest', {'n_estimators': 100, 'max_depth': 10})
        ]
        
        for model_type, params in model_params:
            self.logger.info(f"\n--- Эксперимент: {model_type} с параметрами {params} ---")
            try:
                model = self.create_model(model_type, **params)
                train_score, val_score = self.train_model(model, X_train, y_train, X_val, y_val)
                demo_accuracy, _ = self.test_model(model, X_demo, y_demo, save_predictions=False,title=f"{model_type}_base")
                
                experiments.append({
                    'experiment': f'{model_type}_{str(params)}',
                    'model_type': model_type,
                    'params': params,
                    'train_score': train_score,
                    'val_score': val_score,
                    'demo_accuracy': demo_accuracy,
                    'has_probabilities': self.has_probabilities
                })
            except Exception as e:
                self.logger.error(f"Ошибка в эксперименте {model_type}: {str(e)}")
        
        for C in [0.1, 1.0, 10.0]:
            self.logger.info(f"\n--- Эксперимент: LogisticRegression C={C} ---")
            model = self.create_model('logistic', C=C, max_iter=10000)
            train_score, val_score = self.train_model(model, X_train, y_train, X_val, y_val)
            demo_accuracy, _ = self.test_model(model, X_demo, y_demo, save_predictions=False,title=f"logistic_C{C}")
            
            experiments.append({
                'experiment': f'logistic_C_{C}',
                'model_type': 'logistic',
                'params': {'C': C, 'max_iter': 10000},
                'train_score': train_score,
                'val_score': val_score,
                'demo_accuracy': demo_accuracy,
                'has_probabilities': True
            })
        
        for kernel in ['linear', 'rbf']:
            self.logger.info(f"\n--- Эксперимент: SVM kernel={kernel} ---")
            model = self.create_model('svm', C=1.0, kernel=kernel, probability=True)
            train_score, val_score = self.train_model(model, X_train, y_train, X_val, y_val)
            demo_accuracy, _ = self.test_model(model, X_demo, y_demo, save_predictions=False,title=kernel)
            
            experiments.append({
                'experiment': f'svm_kernel_{kernel}',
                'model_type': 'svm',
                'params': {'C': 1.0, 'kernel': kernel, 'probability': True},
                'train_score': train_score,
                'val_score': val_score,
                'demo_accuracy': demo_accuracy,
                'has_probabilities': True
            })
        
        results_df = pd.DataFrame(experiments)
        results_file = os.path.join(self.output_dir, 'hyperparameter_tuning_results.txt')
        results_df.to_csv(results_file, index=False, sep='\t', encoding='utf-8')
        
        best_experiment = results_df.loc[results_df['demo_accuracy'].idxmax()]
        self.logger.info(f"\n=== ЛУЧШАЯ МОДЕЛЬ ===")
        self.logger.info(f"Эксперимент: {best_experiment['experiment']}")
        self.logger.info(f"Точность: {best_experiment['demo_accuracy']:.4f}")
        self.logger.info(f"Параметры: {best_experiment['params']}")
        
        self.logger.info("\nПереобучение лучшей модели на всех данных...")
        X_all_train = X_train + X_val
        y_all_train = y_train + y_val
        
        best_model = self.create_model(
            best_experiment['model_type'], 
            **best_experiment['params']
        )
        best_model.fit(X_all_train, y_all_train)
        
        self.has_probabilities = best_experiment['has_probabilities']
        
        final_accuracy, y_pred = self.test_model(best_model, X_demo, y_demo,title=f"best_{best_experiment['model_type']}")
        
        self.save_model(best_model, 'best_model.pkl')
        
        self.save_hyperparameter_analysis(results_df)
        
        return best_model, final_accuracy, results_df
    
    def save_model(self, model, filename):
        """Сохранение модели"""
        model_path = os.path.join(self.output_dir, filename)
        joblib.dump({
            'model': model,
            'has_probabilities': self.has_probabilities
        }, model_path)
        self.logger.info(f"Модель сохранена в: {model_path}")
        self.best_model = model
    
    def load_model(self, filename):
        """Загрузка модели"""
        model_path = os.path.join(self.output_dir, filename)
        if os.path.exists(model_path):
            saved_data = joblib.load(model_path)
            self.best_model = saved_data['model']
            self.has_probabilities = saved_data.get('has_probabilities', False)
            self.logger.info(f"Модель загружена из: {model_path}")
            self.logger.info(f"Модель поддерживает вероятности: {self.has_probabilities}")
            return True
        else:
            self.logger.error(f"Файл модели не найден: {model_path}")
            return False
    
    def save_hyperparameter_analysis(self, results_df):
        """Сохранение анализа влияния гиперпараметров"""
        analysis_file = os.path.join(self.output_dir, 'hyperparameter_analysis.txt')
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("АНАЛИЗ ВЛИЯНИЯ ГИПЕРПАРАМЕТРОВ\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. СРАВНЕНИЕ ТИПОВ МОДЕЛЕЙ:\n")
            model_stats = results_df.groupby('model_type')['demo_accuracy'].agg(['mean', 'max', 'min'])
            for model_type, stats in model_stats.iterrows():
                f.write(f"   {model_type}: средняя {stats['mean']:.4f}, "
                       f"максимальная {stats['max']:.4f}, минимальная {stats['min']:.4f}\n")
            
            f.write("\n2. ВЛИЯНИЕ ПАРАМЕТРА C (LogisticRegression):\n")
            logistic_exps = results_df[results_df['model_type'] == 'logistic']
            for _, exp in logistic_exps.iterrows():
                if 'C' in exp['params']:
                    f.write(f"   C={exp['params']['C']}: точность {exp['demo_accuracy']:.4f}\n")
            
            f.write("\n3. ВЛИЯНИЕ ЯДРА (SVM):\n")
            svm_exps = results_df[results_df['model_type'] == 'svm']
            for _, exp in svm_exps.iterrows():
                if 'kernel' in exp['params']:
                    prob_info = " (с вероятностями)" if exp['params'].get('probability', False) else " (без вероятностей)"
                    f.write(f"   kernel={exp['params']['kernel']}{prob_info}: точность {exp['demo_accuracy']:.4f}\n")
            
            f.write("\n4. ВЫВОДЫ:\n")
            best_model = results_df.loc[results_df['demo_accuracy'].idxmax()]
            f.write(f"   Лучшая модель: {best_model['experiment']}\n")
            f.write(f"   Лучшая точность: {best_model['demo_accuracy']:.4f}\n")
            f.write(f"   Поддерживает вероятности: {best_model['has_probabilities']}\n")
            f.write(f"   Рекомендуемые параметры: {best_model['params']}\n")
        
        self.logger.info(f"Анализ гиперпараметров сохранен в: {analysis_file}")
    
    def predict_sentiment(self, text):
        """Предсказание sentiment для нового текста"""
        if self.best_model is None:
            self.logger.error("Модель не загружена!")
            return None
        
        processed_text = ' '.join(self.analyzer.preprocess_text(text))
        prediction = self.best_model.predict([processed_text])[0]
        
        if self.has_probabilities and hasattr(self.best_model, 'predict_proba'):
            probability = self.best_model.predict_proba([processed_text])[0]
            confidence = probability[prediction]
            probabilities = {
                'negative': probability[0],
                'positive': probability[1]
            }
        else:
            confidence = 1.0
            probabilities = {
                'negative': 0.0,
                'positive': 0.0
            }
            
            if hasattr(self.best_model, 'decision_function'):
                try:
                    decision_scores = self.best_model.decision_function([processed_text])
                    if len(decision_scores.shape) == 1:
                        confidence = abs(decision_scores[0])
                        confidence = 1 / (1 + np.exp(-confidence))
                    else:
                        confidence = np.max(decision_scores[0])
                except:
                    confidence = 1.0
        
        sentiment = "Положительный" if prediction == 1 else "Отрицательный"
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': probabilities,
            'has_probabilities': self.has_probabilities
        }

def run_full_experiment():
    """Запуск полного эксперимента"""
    classifier = TextClassifier()
    
    try:
        texts, labels = classifier.load_and_preprocess_data()
        
        processed_texts = classifier.preprocess_texts(texts)
        
        X_train, X_val, X_demo, y_train, y_val, y_demo = classifier.split_dataset(
            processed_texts, labels
        )
        
        best_model, final_accuracy, results = classifier.hyperparameter_tuning(
            X_train, y_train, X_val, y_val, X_demo, y_demo
        )
        
        classifier.logger.info(f"\n=== ЭКСПЕРИМЕНТ ЗАВЕРШЕН ===")
        classifier.logger.info(f"Финальная точность: {final_accuracy:.4f}")
        classifier.logger.info(f"Модель поддерживает вероятности: {classifier.has_probabilities}")
        
        return classifier
        
    except Exception as e:
        classifier.logger.error(f"Ошибка в эксперименте: {str(e)}")
        raise

if __name__ == "__main__":
    run_full_experiment()