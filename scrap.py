from playwright.sync_api import sync_playwright
import random
import time
import json
import csv
from datetime import datetime
import re
from math import ceil
import os

class KinopoiskReviewScraper:
    def __init__(self, headless=True, slow_mode=True):
        self.headless = headless
        self.slow_mode = slow_mode
        self.reviews_data = []
        self.failed_requests = 0
        self.max_retries = 3
        
        self.file_counters = {'good': 0, 'bad': 184}
        
        os.makedirs('dataset/good', exist_ok=True)
        os.makedirs('dataset/bad', exist_ok=True)
        
    def human_like_behavior(self, page):
        viewport = page.viewport_size
        if viewport:
            for _ in range(random.randint(2, 5)):
                x = random.randint(100, viewport['width'] - 100)
                y = random.randint(100, viewport['height'] - 100)
                page.mouse.move(x, y)
                time.sleep(0.1)
        
        for _ in range(random.randint(1, 3)):
            page.evaluate("window.scrollBy(0, Math.random() * 300)")
            time.sleep(0.5)

    def setup_stealth(self, context):
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );

            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3],
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['ru-RU', 'ru', 'en-US', 'en'],
            });
        """)

    def wait_and_solve_captcha(self, page):
        captcha_selector = 'div.CheckboxCaptcha'
        
        try:
            page.wait_for_selector(captcha_selector, timeout=10000)
            print("Обнаружена капча")
            
            if not self.headless:
                print("Решите капчу вручную...")
                page.wait_for_selector(captcha_selector, state='hidden', timeout=120000)
                print("Капча решена")
                return True
            else:
                return self.auto_solve_captcha(page)
                
        except Exception:
            return True

    def auto_solve_captcha(self, page):
        try:
            checkbox = page.query_selector('input.CheckboxCaptcha-Button')
            if checkbox:
                checkbox.click()
                time.sleep(3)
                
                captcha = page.query_selector('div.CheckboxCaptcha')
                if not captcha:
                    print("Капча пройдена")
                    return True
                    
            print("Не удалось решить капчу")
            return False
            
        except Exception as e:
            print(f"Ошибка капчи: {e}")
            return False

    def get_total_pages(self, page, per_page):
        try:
            pages_info_element = page.query_selector('div.pagesFromTo')
            
            if pages_info_element:
                pages_text = pages_info_element.inner_text().strip()
                match = re.search(r'из\s+(\d+)', pages_text)
                if match:
                    total_reviews = int(match.group(1))
                    total_pages = ceil(total_reviews / per_page)
                    return total_pages
            
            pagination_links = page.query_selector_all('div.paginator a.paginator__page')
            if pagination_links:
                last_page = int(pagination_links[-1].inner_text())
                return last_page
            
            return 1
            
        except Exception as e:
            print(f"Ошибка определения страниц: {e}")
            return 1

    def wait_for_reviews_load(self, page):
        try:
            page.wait_for_selector('.reviewItem', timeout=30000)
            
            page.wait_for_function(
                """() => {
                    const reviews = document.querySelectorAll('.reviewItem');
                    const hasContent = document.querySelector('.brand_words, span[itemprop="reviewBody"]') !== null;
                    const hasPagination = document.querySelector('div.pagesFromTo') !== null || reviews.length > 0;
                    return hasContent && hasPagination;
                }""",
                timeout=20000
            )
            return True
        except Exception as e:
            return page.query_selector('.reviewItem') is not None

    def get_film_title(self, page):
        try:
            title_element = page.query_selector('div.breadcrumbs__sub')
            if title_element:
                return title_element.inner_text().strip()
            
            title_element = page.query_selector('span[itemprop="name"]')
            if title_element:
                return title_element.inner_text().strip()
            
            return "Неизвестный фильм"
        except Exception as e:
            print(f"Ошибка получения названия фильма: {e}")
            return "Неизвестный фильм"

    def save_review_to_file(self, review_data, film_title, status):
        try:
            counter = self.file_counters[status]
            
            filename = f"{counter:04d}.txt"
            filepath = f"dataset/{status}/{filename}"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(film_title + '\n')
                f.write(review_data['content'] + '\n')
            
            print(f"Сохранен отзыв: {filepath}")
            
            self.file_counters[status] += 1
            
            return True
        except Exception as e:
            print(f"Ошибка сохранения отзыва в файл: {e}")
            return False

    def extract_review_data(self, review_element):
        try:
            review_id = review_element.get_attribute('data-id') or ''
            
            user_element = review_element.query_selector('.profile_name a[itemprop="name"]')
            username = user_element.inner_text().strip() if user_element else "Анонимный пользователь"
            
            content_element = review_element.query_selector('span[itemprop="reviewBody"]')
            if not content_element:
                content_element = review_element.query_selector('.brand_words')
            content = content_element.inner_text().strip() if content_element else ""
            
            date_element = review_element.query_selector('.date')
            date = date_element.inner_text().strip() if date_element else ""
            
            review_class = review_element.get_attribute('class') or ''
            if 'bad' in review_class:
                review_type = "negative"
            elif 'good' in review_class:
                review_type = "positive"
            else:
                review_type = "neutral"
            
            vote_text = ""
            if review_id:
                vote_element = review_element.query_selector(f'#comment_num_vote_{review_id}')
                if vote_element:
                    vote_text = vote_element.inner_text().strip()
            
            if not vote_text:
                vote_element = review_element.query_selector('ul.useful li:has-text("/")')
                if vote_element:
                    vote_text = vote_element.inner_text().strip()
            
            if '/' in vote_text:
                try:
                    parts = vote_text.split('/')
                    useful = int(parts[0].strip()) if parts[0].strip().isdigit() else 0
                    not_useful = int(parts[1].strip()) if parts[1].strip().isdigit() else 0
                except (ValueError, IndexError):
                    useful, not_useful = 0, 0
            else:
                useful, not_useful = 0, 0
            
            return {
                'review_id': review_id,
                'username': username,
                'content': content,
                'date': date,
                'type': review_type,
                'useful_votes': useful,
                'not_useful_votes': not_useful,
                'scraped_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"Ошибка извлечения: {e}")
            return None

    def get_browser_context(self, p):
        launch_options = {
            'headless': self.headless,
            'args': [
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-web-security',
                f'--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(120, 125)}.0.0.0 Safari/537.36'
            ]
        }
        
        browser = p.chromium.launch(**launch_options)
        
        context_options = {
            'viewport': {'width': 1920, 'height': 1080},
            'user_agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(120, 125)}.0.0.0 Safari/537.36',
            'locale': 'ru-RU',
            'timezone_id': 'Europe/Moscow'
        }
        
        context = browser.new_context(**context_options)
        self.setup_stealth(context)
        
        return browser, context

    def safe_navigate(self, page, url):
        try:
            response = page.goto(url, wait_until='networkidle', timeout=60000)
            
            if response and response.status >= 400:
                print(f"HTTP ошибка: {response.status}")
                return False
            
            if not self.wait_and_solve_captcha(page):
                return False
                
            return True
            
        except Exception as e:
            print(f"Ошибка навигации: {e}")
            return False

    def process_single_page(self, page, film_id, status, page_num, per_page, base_url, film_title):
        review_elements = page.query_selector_all('.reviewItem')
        page_reviews_count = len(review_elements)
        
        for i, review_element in enumerate(review_elements):
            review_data = self.extract_review_data(review_element)
            if review_data:
                review_data.update({
                    'film_id': film_id,
                    'status': status,
                    'page': page_num,
                    'per_page': per_page,
                    'url': base_url + f"{page_num}/"
                })
                self.reviews_data.append(review_data)
                
                self.save_review_to_file(review_data, film_title, status)
        
        print(f"Страница {page_num}: {page_reviews_count} отзывов")
        return page_reviews_count

    def process_all_pages(self, page, film_id, status, per_page, total_pages, base_url):
        total_reviews_collected = 0
        
        film_title = self.get_film_title(page)
        print(f"Название фильма: {film_title}")
        
        for page_num in range(1, total_pages + 1):
            if page_num > 1:
                page_url = base_url + f"{page_num}/"
                print(f"Переходим на страницу {page_num}")
                
                if not self.safe_navigate(page, page_url):
                    print(f"Не удалось загрузить страницу {page_num}")
                    break
                
                if self.slow_mode:
                    self.human_like_behavior(page)
                    time.sleep(random.uniform(1, 3))
            
            page_reviews = self.process_single_page(page, film_id, status, page_num, per_page, base_url, film_title)
            total_reviews_collected += page_reviews
            
            if page_num < total_pages:
                delay = random.uniform(3, 6) if self.slow_mode else random.uniform(1, 3)
                time.sleep(delay)
        
        return total_reviews_collected

    def scrape_film_reviews_with_retry(self, film_id, status='good', per_page=20, retry_count=0):
        if retry_count >= self.max_retries:
            print(f"Лимит попыток для фильма {film_id}")
            return 0
            
        base_url = f"https://www.kinopoisk.ru/film/{film_id}/reviews/ord/date/status/{status}/perpage/{per_page}/page/"
        
        with sync_playwright() as p:
            browser, context = self.get_browser_context(p)
            page = context.new_page()
            
            try:
                first_page_url = base_url + '1/'
                print(f"Загружаем: {first_page_url} (попытка {retry_count + 1})")
                
                navigation_success = self.safe_navigate(page, first_page_url)
                if not navigation_success:
                    return self.scrape_film_reviews_with_retry(film_id, status, per_page, retry_count + 1)
                
                if self.slow_mode:
                    self.human_like_behavior(page)
                    time.sleep(random.uniform(2, 4))
                
                if not self.wait_for_reviews_load(page):
                    return 0
                
                if not self.wait_and_solve_captcha(page):
                    return self.scrape_film_reviews_with_retry(film_id, status, per_page, retry_count + 1)
                
                total_pages = self.get_total_pages(page, per_page)
                if total_pages == 0:
                    return 0
                
                print(f"Всего страниц: {total_pages}")
                return self.process_all_pages(page, film_id, status, per_page, total_pages, base_url)
                
            except Exception as e:
                print(f"Ошибка сбора: {e}")
                return 0
            finally:
                browser.close()

    def scrape_multiple_films(self, film_ids, statuses=['good', 'bad'], per_page=20):
        total_reviews = 0
        
        for film_id in film_ids:
            for status in statuses:
                try:
                    print(f"\nОбрабатываем фильм ID: {film_id}, статус: {status}")
                    
                    reviews_count = self.scrape_film_reviews_with_retry(film_id, status, per_page)
                    total_reviews += reviews_count
                    
                    print(f"Фильм {film_id} ({status}): {reviews_count} отзывов")
                    
                except Exception as e:
                    print(f"Ошибка обработки: {e}")
                
                delay = random.uniform(4, 8)
                time.sleep(delay)
        
        return total_reviews

    def get_stats(self):
        total_reviews = len(self.reviews_data)
        positive_reviews = len([r for r in self.reviews_data if r['type'] == 'positive'])
        negative_reviews = len([r for r in self.reviews_data if r['type'] == 'negative'])
        neutral_reviews = len([r for r in self.reviews_data if r['type'] == 'neutral'])
        unique_films = len(set(r['film_id'] for r in self.reviews_data))
        
        return {
            'total_reviews': total_reviews,
            'positive_reviews': positive_reviews,
            'negative_reviews': negative_reviews,
            'neutral_reviews': neutral_reviews,
            'unique_films': unique_films
        }

def main():
    scraper = KinopoiskReviewScraper(headless=False, slow_mode=True)
    
    try:
        film_ids = list(range(350,500))
        
        total_reviews = scraper.scrape_multiple_films(
            film_ids=film_ids,
            statuses=['bad'],
            per_page=50
        )
        
        stats = scraper.get_stats()
        print("\nСтатистика:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        scraper.save_to_json()
        scraper.save_to_csv()
        
        print(f"\nСохранено файлов с положительными отзывами: {scraper.file_counters['good']}")
        print(f"Сохранено файлов с отрицательными отзывами: {scraper.file_counters['bad']}")
        
    except KeyboardInterrupt:
        print("Прервано пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()