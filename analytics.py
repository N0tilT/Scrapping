import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

class TextAnalyzer:
    def __init__(self, output_dir='output_plots', threshold=1000):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.output_dir = output_dir
        self.threshold = threshold

        os.makedirs(self.output_dir, exist_ok=True)
        self.stop_words = {
            'весь', 'свой','фильм', 'это', 'который', 'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только',
            'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
            'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять',
            'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо',
            'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже',
            'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним',
            'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех',
            'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот',
            'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем',
            'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более',
            'всегда', 'конечно', 'всю', 'между'
        }

    def _parse_movie_info(self, text):
        """
        Извлекает название фильма и год из текста отзыва.
        """
        lines = text.strip().split('\n')
        if len(lines) > 0:
            first_line = lines[0].strip()
            year_match = re.search(r'\((\d{4})\)|,?\s*(\d{4})$', first_line)
            if year_match:
                year = year_match.group(1) or year_match.group(2)
                title = first_line[:year_match.start()].strip().rstrip(',')
                return title, int(year)
        return "Неизвестный фильм", None

    def _save_plot(self, filename, dpi=300, bbox_inches='tight'):
        """Сохраняет текущий график в файл"""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
        plt.close()

    def load_reviews_from_folders(self, good_path='dataset/good', bad_path='dataset/bad'):
        """
        Загрузка отзывов из папок good и bad
        
        Returns:
            pd.DataFrame: DataFrame с колонками ['review_type', 'text', 'movie_title', 'year']
        """
        records = []
        
        if os.path.exists(good_path):
            for filename in os.listdir(good_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(good_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read().strip()
                    if text:
                        movie_title, year = self._parse_movie_info(text)
                        records.append({
                            'review_type': 'good',
                            'text': text,
                            'file_name': filename,
                            'movie_title': movie_title,
                            'year': year
                        })
                        if len(records) >= self.threshold: 
                            break
                    
        if os.path.exists(bad_path):
            for filename in os.listdir(bad_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(bad_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read().strip()
                    if text:
                        movie_title, year = self._parse_movie_info(text)
                        records.append({
                            'review_type': 'bad', 
                            'text': text,
                            'file_name': filename,
                            'movie_title': movie_title,
                            'year': year
                        })
                        if len(records) >= self.threshold * 2: 
                            break

        df = pd.DataFrame(records)
        return df

    def preprocess_dataframe(self, df):
        """
        Предобработка DataFrame с добавлением информации о фильмах.
        """
        if df.empty:
            return df
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        print("ПРОВЕРКА НА НЕВАЛИДНЫЕ ЗНАЧЕНИЯ:")
        print(f"Всего строк: {len(df)}")
        print(f"Невалидные значения по колонкам:")
        print(df.isnull().sum())
        print(f"Пустые тексты: {df['text'].isnull().sum()}")
        
        initial_count = len(df)
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip() != '']
        removed_count = initial_count - len(df)
        print(f"Удалено строк: {removed_count}")
        
        df['word_count'] = df['text'].apply(self._count_words)
        df['class_label'] = df['review_type'].map({'good': 1, 'bad': 0})
        
        return df

    def _count_words(self, text):
        """Подсчет количества слов в тексте с использованием Natasha"""
        doc = Doc(text)
        doc.segment(self.segmenter)
        return len(doc.tokens)

    def get_text_statistics(self, df):
        """Вычисление статистической информации для числовых колонок"""
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            print(f"\nСтатистика для '{col}':")
            print(f"  Мин: {df[col].min()}")
            print(f"  Макс: {df[col].max()}")
            print(f"  Среднее: {df[col].mean():.2f}")
            print(f"  Медиана: {df[col].median()}")
            print(f"  Стандартное отклонение: {df[col].std():.2f}")
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"  Выбросы: {len(outliers)} строк")
            print(f"  Квантили:")
            quantiles = df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            for q, val in quantiles.items():
                print(f"    {int(q*100)}%: {val:.2f}")

    def filter_by_word_count(self, df, max_words):
        """
        Фильтрация DataFrame по максимальному количеству слов
        """
        filtered_df = df[df['word_count'] <= max_words].sort_values('word_count')
        print(f"\nФильтрация по макс. {max_words} слов:")
        print(f"Осталось строк: {len(filtered_df)} из {len(df)}")
        return filtered_df

    def filter_by_class_label(self, df, class_label):
        """
        Фильтрация DataFrame по метке класса
        """
        filtered_df = df[df['class_label'] == class_label]
        label_name = 'positive' if class_label == 1 else 'negative'
        print(f"\nФильтрация по классу: {label_name}")
        print(f"Найдено строк: {len(filtered_df)}")
        return filtered_df

    def filter_by_year(self, df, year):
        """
        Фильтрация DataFrame по году выпуска фильма
        """
        filtered_df = df[df['year'] == year]
        print(f"\nФильтрация по году: {year}")
        print(f"Найдено строк: {len(filtered_df)}")
        return filtered_df

    def filter_by_movie(self, df, movie_title):
        """
        Фильтрация DataFrame по названию фильма
        """
        filtered_df = df[df['movie_title'] == movie_title]
        print(f"\nФильтрация по фильму: {movie_title}")
        print(f"Найдено строк: {len(filtered_df)}")
        return filtered_df

    def calculate_group_statistics(self, df):
        """
        Группировка по метке класса с вычислением статистики по количеству слов
        """
        group_stats = df.groupby('review_type')['word_count'].agg([
            ('min_count', 'min'),
            ('max_count', 'max'), 
            ('mean_count', 'mean'),
            ('median_count', 'median'),
            ('total_reviews', 'count')
        ]).round(2)
        
        print("Статистика по группам:")
        print(group_stats)
        return group_stats

    def calculate_year_statistics(self, df):
        """
        Статистика по годам выпуска фильмов
        """
        year_stats = df.groupby('year').agg({
            'review_type': 'count',
            'class_label': 'mean',
            'word_count': 'mean'
        }).round(2)
        
        year_stats.columns = ['total_reviews', 'positive_ratio', 'avg_word_count']
        print("Статистика по годам:")
        print(year_stats)
        return year_stats

    def calculate_movie_statistics(self, df):
        """
        Статистика по фильмам
        """
        movie_stats = df.groupby('movie_title').agg({
            'review_type': 'count',
            'class_label': 'mean',
            'word_count': 'mean',
            'year': 'first'
        }).round(2)
        
        movie_stats.columns = ['total_reviews', 'positive_ratio', 'avg_word_count', 'year']
        print("Статистика по фильмам:")
        print(movie_stats.sort_values('total_reviews', ascending=False).head(10))
        return movie_stats

    def preprocess_text(self, text):
        """
        Предобработка текста: токенизация, лемматизация, очистка
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        
        clean_lemmas = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemma = token.lemma.lower()
            
            if (lemma not in self.stop_words and 
                len(lemma) > 2 and 
                lemma.isalpha() and
                not re.match(r'^[а-яё]{1,2}$', lemma)):
                clean_lemmas.append(lemma)
                
        return clean_lemmas

    def create_word_histogram(self, df, class_label, top_n=20):
        """
        Создание гистограммы частоты слов для заданного класса
        """
        class_name = 'positive' if class_label == 1 else 'negative'
        df_filtered = df[df['class_label'] == class_label]
        
        if df_filtered.empty:
            print(f"Нет данных для класса: {class_name}")
            return [], []
        
        all_lemmas = []
        for text in df_filtered['text']:
            lemmas = self.preprocess_text(text)
            all_lemmas.extend(lemmas)
        
        word_freq = Counter(all_lemmas)
        common_words = word_freq.most_common(top_n)
        
        if not common_words:
            print(f"Нет слов для отображения для класса: {class_name}")
            return [], []
            
        words, counts = zip(*common_words)
        
        print(f"\nТоп-{top_n} слов для класса '{class_name}':")
        for i, (word, count) in enumerate(common_words, 1):
            print(f"{i:2d}. {word}: {count}")
        
        return words, counts

    def visualize_histogram(self, words, counts, class_label, title_suffix=""):
        """
        Визуализация гистограммы частоты слов
        """
        if not words or not counts:
            print("Нет данных для визуализации")
            return
        
        class_name = 'Положительные' if class_label == 1 else 'Отрицательные'
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(words, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=10)
        
        plt.title(f'{len(words)} самых частых слов: {class_name} отзывы {title_suffix}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Слова (леммы)', fontsize=12)
        plt.ylabel('Частота', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        filename = f"word_histogram_{class_name}.png"
        self._save_plot(filename)

    def visualize_year_statistics(self, df):
        """
        Визуализация статистики по годам
        """
        year_stats = self.calculate_year_statistics(df)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.bar(year_stats.index, year_stats['total_reviews'], color='lightblue')
        plt.title('Количество отзывов по годам')
        plt.xlabel('Год')
        plt.ylabel('Количество отзывов')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.bar(year_stats.index, year_stats['positive_ratio'], color='lightgreen')
        plt.title('Доля положительных отзывов по годам')
        plt.xlabel('Год')
        plt.ylabel('Доля положительных отзывов')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        plt.bar(year_stats.index, year_stats['avg_word_count'], color='lightcoral')
        plt.title('Средняя длина отзывов по годам')
        plt.xlabel('Год')
        plt.ylabel('Среднее количество слов')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self._save_plot('year_statistics.png')
        

    def visualize_movie_statistics(self, df, top_n=15):
        """
        Визуализация статистики по фильмам
        """
        movie_stats = self.calculate_movie_statistics(df)
        
        if movie_stats.empty:
            print("Нет данных для визуализации по фильмам")
            return
            
        top_movies = movie_stats.nlargest(top_n, 'total_reviews')
        
        plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 2, 1)
        plt.barh(range(len(top_movies)), top_movies['total_reviews'], 
                color='skyblue', alpha=0.7)
        plt.title(f'Топ-{top_n} фильмов по количеству отзывов', fontsize=14, fontweight='bold')
        plt.xlabel('Количество отзывов')
        plt.yticks(range(len(top_movies)), top_movies.index, fontsize=9)
        
        for i, v in enumerate(top_movies['total_reviews']):
            plt.text(v + 0.1, i, str(v), va='center', fontsize=9)
        
        plt.subplot(2, 2, 2)
        colors = ['lightcoral' if x < 0.5 else 'lightgreen' for x in top_movies['positive_ratio']]
        plt.barh(range(len(top_movies)), top_movies['positive_ratio'], 
                color=colors, alpha=0.7)
        plt.title('Доля положительных отзывов по фильмам', fontsize=14, fontweight='bold')
        plt.xlabel('Доля положительных отзывов')
        plt.yticks(range(len(top_movies)), top_movies.index, fontsize=9)
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        
        for i, v in enumerate(top_movies['positive_ratio']):
            plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=9)
        
        plt.subplot(2, 2, 3)
        plt.barh(range(len(top_movies)), top_movies['avg_word_count'], 
                color='purple', alpha=0.7)
        plt.title('Средняя длина отзывов по фильмам', fontsize=14, fontweight='bold')
        plt.xlabel('Среднее количество слов')
        plt.yticks(range(len(top_movies)), top_movies.index, fontsize=9)
        
        for i, v in enumerate(top_movies['avg_word_count']):
            plt.text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=9)
        
        plt.subplot(2, 2, 4)
        movie_years = df[df['movie_title'].isin(top_movies.index)]
        movie_sentiment = movie_years.groupby(['movie_title', 'review_type']).size().unstack(fill_value=0)
        movie_sentiment = movie_sentiment.reindex(top_movies.index)
        movie_sentiment.plot(kind='barh', stacked=True, ax=plt.gca(), 
                           color=['lightcoral', 'lightgreen'], alpha=0.7)
        plt.title('Распределение отзывов по фильмам и типам', fontsize=14, fontweight='bold')
        plt.xlabel('Количество отзывов')
        plt.legend(['Отрицательные', 'Положительные'], fontsize=10)
        
        plt.tight_layout()
        self._save_plot('movie_statistics.png')

    def analyze_texts_comprehensive(self, good_path='dataset/good', bad_path='dataset/bad'):
        """
        Полный анализ текстов с добавлением статистики по годам и фильмам
        """
        df = self.load_reviews_from_folders(good_path, bad_path)
        print(f"Загружено отзывов: {len(df)}")
        print(f"Распределение по классам:")
        print(df['review_type'].value_counts())
        
        df = self.preprocess_dataframe(df)
        
        print("ИНФОРМАЦИЯ О ДАННЫХ:")
        print(f"Колонки: {list(df.columns)}")
        print(f"Размер данных: {df.shape}")
        self.get_text_statistics(df)
        
        print("ФИЛЬТРАЦИЯ:")
        filtered_by_words = self.filter_by_word_count(df, max_words=100)
        if len(filtered_by_words) > 0:
            print(f"Пример отфильтрованного текста ({filtered_by_words.iloc[0]['word_count']} слов):")
            print(filtered_by_words.iloc[0]['text'][:200] + "...")
        
        positive_reviews = self.filter_by_class_label(df, class_label=1)
        negative_reviews = self.filter_by_class_label(df, class_label=0)
        group_stats = self.calculate_group_statistics(df)
        
        print("СТАТИСТИКА ПО ГОДАМ И ФИЛЬМАМ:")
        year_stats = self.calculate_year_statistics(df)
        movie_stats = self.calculate_movie_statistics(df)
        self.visualize_year_statistics(df)
        self.visualize_movie_statistics(df)
        
        print("ЧАСТОТА СЛОВ:")
        pos_words, pos_counts = self.create_word_histogram(df, class_label=1, top_n=15)
        if pos_words and pos_counts:
            self.visualize_histogram(pos_words, pos_counts, class_label=1, 
                                   title_suffix="(положительные)")
        neg_words, neg_counts = self.create_word_histogram(df, class_label=0, top_n=15)
        if neg_words and neg_counts:
            self.visualize_histogram(neg_words, neg_counts, class_label=0,
                                   title_suffix="(отрицательные)")
        self._create_additional_visualizations(df)
        
        return df

    def _create_additional_visualizations(self, df):
        """Дополнительные визуализации с сохранением в файлы"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='review_type', y='word_count')
        plt.title('Распределение количества слов по классам')
        plt.xlabel('Тип отзыва')
        plt.ylabel('Количество слов')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=df, x='word_count', hue='review_type', bins=20, alpha=0.6)
        plt.title('Гистограмма распределения количества слов')
        plt.xlabel('Количество слов')
        plt.ylabel('Частота')
        
        plt.tight_layout()
        self._save_plot('word_distribution_by_class.png')
        
        plt.figure(figsize=(8, 6))
        review_counts = df['review_type'].value_counts()
        colors = ['lightgreen', 'lightcoral']
        plt.pie(review_counts.values, labels=review_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Распределение отзывов по типам')
        plt.tight_layout()
        self._save_plot('reviews_distribution_pie.png')
        
        plt.figure(figsize=(10, 6))
        group_stats = df.groupby('review_type')['word_count'].agg(['mean', 'median']).round(2)
        
        x = range(len(group_stats))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], group_stats['mean'], width, label='Среднее', alpha=0.7)
        plt.bar([i + width/2 for i in x], group_stats['median'], width, label='Медиана', alpha=0.7)
        
        plt.xlabel('Тип отзыва')
        plt.ylabel('Количество слов')
        plt.title('Среднее и медианное количество слов по типам отзывов')
        plt.xticks(x, group_stats.index)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save_plot('word_stats_by_class.png')

def main():
    analyzer = TextAnalyzer(threshold=5000)
    
    try:
        df = analyzer.analyze_texts_comprehensive(
            good_path='dataset/good',
            bad_path='dataset/bad'
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()