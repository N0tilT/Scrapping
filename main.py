from bs4 import BeautifulSoup
import requests
import time
from random import uniform, choice
import logging
from urllib.parse import urljoin
import re
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

review_counters = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]

REFERERS = [
    'https://www.google.com/',
    'https://www.yandex.ru/',
    'https://www.bing.com/',
    'https://duckduckgo.com/'
]

BASE_URL = "https://otzovik.com/reviews/market_yandex_ru-yandeks_market/"

def get_session():
    session = requests.Session()
    session.headers.update({
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    })
    return session

def get_headers():
    return {
        'User-Agent': choice(USER_AGENTS),
        'Referer': choice(REFERERS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    }

def process_page(session, page, ratio):
    url = f"{BASE_URL}{page}/?ratio={ratio}"
    
    try:
        delay = uniform(3, 8)
        logger.info(f"Жду {delay:.2f} секунд перед запросом")
        time.sleep(delay)
        
        headers = get_headers()
        response = session.get(url, headers=headers, timeout=15)
        
        
        if response.status_code != 200:
            logger.warning(f"Ошибка {response.status_code} на странице {page}")
            return False
        
        soup = BeautifulSoup(response.text, 'html.parser')
        review_divs = soup.find_all('div', itemprop='review')
        
        if not review_divs:
            logger.info(f"Отзывы не найдены на странице {page}")
            return False

        logger.info(f"Страница {page}, найдено отзывов: {len(review_divs)}")
        
        for review in review_divs:
            process_review(session, review, ratio)
            
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка сети: {e}")
        time.sleep(uniform(10, 20))
        return False

def process_review(session, review, ratio):
    try:
        review_link = None
        
        meta_url = review.find('meta', itemprop='url')
        if meta_url:
            review_link = meta_url.get('content')
        
        if not review_link:
            logger.warning("Не удалось найти ссылку на отзыв")
            return

        time.sleep(uniform(2, 5))
        
        headers = get_headers()
        response_review = session.get(review_link, headers=headers, timeout=15)
        
        if response_review.status_code != 200:
            logger.warning(f"Ошибка при получении отзыва: {response_review.status_code}")
            return

        review_soup = BeautifulSoup(response_review.text, 'html.parser')
        
        title = review_soup.find('h1')
        title_text = title.get_text(strip=True) if title else "Не найдено"
        
        plus_div = review_soup.find('div', class_='review-plus')
        plus_text = plus_div.get_text(strip=True).replace('Достоинства:', '').strip() if plus_div else "Не указано"
        
        minus_div = review_soup.find('div', class_='review-minus')
        minus_text = minus_div.get_text(strip=True).replace('Недостатки:', '').strip() if minus_div else "Не указано"
        
        review_body = review_soup.find('div', class_='review-body')
        review_text = ' '.join(review_body.get_text(strip=True, separator=' ').split()) if review_body else "Не найдено"
        content = f"""Заголовок: {title_text}
                    Достоинства: {plus_text}
                    Недостатки: {minus_text}
                    Текст отзыва: {review_text}
                    """

        ratio_dir = f"dataset/ratio_{ratio}"
        os.makedirs(ratio_dir, exist_ok=True)
        global review_counters
        review_counters[ratio] += 1
        file_number = review_counters[ratio]
        
        filename = f"{file_number:04d}.txt"
        filepath = os.path.join(ratio_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        
        logger.info(f"Отзыв с рейтингом {ratio} сохранен в файл: {filepath}")

    except Exception as e:
        logger.error(f"Ошибка при обработке отзыва: {e}")

def main():
    session = get_session()
    page = 1
    ratio = 1
    consecutive_errors = 0

    while ratio <= 5 and consecutive_errors < 5:
        result = process_page(session, page, ratio)
        
        if result is None:
            consecutive_errors += 1
            logger.info(f"Приостанавливаю работу на 60-120 секунд (ошибка {consecutive_errors}/5)")
            time.sleep(uniform(60, 120))
            continue
            
        if result is False:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                logger.warning("Много последовательных ошибок. Меняю ratio.")
                page = 1
                ratio += 2
                consecutive_errors = 0
                if ratio > 5:
                    break
                continue
            time.sleep(uniform(10, 20))
            continue
            
        consecutive_errors = 0
        page += 1

    logger.info("Парсинг завершен.")

if __name__ == "__main__":
    main()