import requests
from bs4 import BeautifulSoup
import json
import time
import os
import glob
import random

# ScraperAPI anahtarını doğrudan kullan
SCRAPINGBEE_API_KEY = "WFIUDK80YC8AA5FX21FLBEXSVNGKAH3EN0F8J5CZXA4MVMKXV3EIKIGLZRN8CMS8SQ634EX4FVSVQM1V"
API_LIMIT = 790  # Güncel ScrapingBee API limiti

def load_all_restaurant_links_with_names():
    """Tüm ana ve alt klasörlerdeki tüm restoran adlarını ve linklerini sıralı ve tekrarsız olarak döndürür."""
    all_restaurants = []
    seen_links = set()
    json_folder = "Restaurants-g1221508-Kahramanmaras_Kahramanmaras_Province"
    json_files = glob.glob(os.path.join(json_folder, "**", "*.json"), recursive=True)
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    continue
                data = json.loads(content)
                for restaurant in data:
                    link = restaurant.get('link', '').strip()
                    if link and link not in seen_links:
                        name = restaurant.get('name', '').strip()
                        all_restaurants.append({'name': name, 'link': link})
                        seen_links.add(link)
        except Exception as e:
            print(f"Dosya okuma hatası {file_path}: {e}")
    # Numara ekle/temizle
    for idx, rest in enumerate(all_restaurants, 1):
        # Sadece adı numaralandır, varsa baştaki numarayı kaldır
        name = rest['name']
        if '.' in name and name.split('.')[0].isdigit():
            name = name[name.find('.')+1:].strip()
        rest['name'] = f"{idx}. {name}"
    return all_restaurants

def read_existing_restaurants_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def append_or_update_restaurant(existing, new_rest):
    # Aynı url varsa güncelle, yoksa ekle
    for i, r in enumerate(existing):
        if r.get('url') == new_rest.get('url'):
            existing[i] = new_rest
            return existing
    existing.append(new_rest)
    return existing

def fetch_page_scrapingbee(url):
    global API_counter
    if API_counter >= API_LIMIT:
        return None
    params = {
        'api_key': SCRAPINGBEE_API_KEY,
        'url': url,
        'render_js': 'true',
        'block_resources': 'false',
        'wait': '2000',
    }
    response = requests.get('https://app.scrapingbee.com/api/v1/', params=params, timeout=120)
    API_counter += 1
    if response.status_code == 200:
        return response.text
    else:
        print(f"[!] ScrapingBee Hatası: {response.status_code} - {response.text}")
        return None

def scrape_all_reviews(url, total_review_count=None):
    """Tüm sayfalardaki tüm yorumları eksiksiz ve tekrarsız çeker. Sayfa sayısı dinamik olarak hesaplanır."""
    all_comments = []
    seen_comments = set()
    page_num = 0
    # Eğer toplam yorum sayısı biliniyorsa, sayfa sayısını hesapla
    max_pages = None
    if total_review_count:
        max_pages = (total_review_count // 10) + 2  # +1 fazlası da denensin
    else:
        max_pages = 20  # Güvenlik için üst limit (çok büyük restoranlar için)
    while True:
        if page_num == 0:
            page_url = url
        else:
            insert_point = url.find('-Reviews-') + len('-Reviews-')
            page_url = url[:insert_point] + f'or{page_num*10}-' + url[insert_point:]
        params = {
            'api_key': SCRAPINGBEE_API_KEY,
            'url': page_url,
            'render_js': 'true',
            'block_resources': 'false',
            'wait': '2000',
        }
        html = requests.get('https://app.scrapingbee.com/api/v1/', params=params, timeout=120).text
        soup = BeautifulSoup(html, "lxml")
        selectors = [
            'span[data-automation^="reviewText_"]',
            'q.QewHA',
            'q.XllAv',
            'div.review-container q',
            'div.review-content span',
            'span.fullText',
            'q',
        ]
        page_comments = []
        for sel in selectors:
            found = [el.get_text(strip=True) for el in soup.select(sel)]
            for text in found:
                if text and text not in seen_comments:
                    page_comments.append(text)
                    seen_comments.add(text)
        if page_comments:
            all_comments.extend(page_comments)
        # Sonraki sayfa linki var mı kontrol et
        next_btn = soup.select_one('a.ui_button.nav.next.primary, button[aria-label*="Sonraki"], button[aria-label*="Next"]')
        # Son sayfa kontrolü: ya next yok ya da max_pages'e ulaşıldıysa kır
        if (not next_btn and page_num+1 >= max_pages) or (max_pages and page_num+1 >= max_pages):
            break
        page_num += 1
        time.sleep(2)
        # Yeterli yoruma ulaşıldıysa erken çık
        if total_review_count and len(all_comments) >= total_review_count:
            break
    return all_comments

def fetch_total_review_count(soup):
    """Sayfadaki toplam yorum sayısını çeker (TripAdvisor arayüzüne göre)."""
    selectors = [
        'span[data-automation="reviewCount"]',
        'span[data-test-target="reviews-tab"]',
        'a[data-tab="TABS_REVIEWS"] span',
        'span.reviewCount',
        'span[data-automation="reviewCount"]',
    ]
    for sel in selectors:
        tag = soup.select_one(sel)
        if tag:
            text = tag.get_text(strip=True)
            digits = ''.join(filter(str.isdigit, text))
            if digits:
                return int(digits)
    h2s = soup.find_all('h2')
    for h2 in h2s:
        if 'yorum' in h2.text.lower():
            digits = ''.join(filter(str.isdigit, h2.text))
            if digits:
                return int(digits)
    return None

def scrape_restaurant(name, url):
    """Fetches detailed data for a single restaurant page (via ScraperAPI)"""
    try:
        html = fetch_page_scrapingbee(url)
        if html is None:
            print(f"API limiti doldu veya sayfa çekilemedi: {url}")
            return None
        with open("son_html.html", "w", encoding="utf-8") as f:
            f.write(html)
        soup = BeautifulSoup(html, "lxml")
        # Restaurant name
        name_from_html = "Unknown"
        name_tag = soup.find("h1")
        if name_tag:
            name_from_html = name_tag.text.strip()
        # Rating
        rating = None
        rating_tag = soup.find('div', {'data-automation': 'bubbleRatingValue'})
        if rating_tag:
            rating = rating_tag.text.strip()
        # Tüm yorumları çek (sayfa sayısı dinamik, expected_review_count kaldırıldı)
        comments = scrape_all_reviews(url)
        return {
            "restaurant_name": name_from_html if name_from_html != "Unknown" else name,
            "rating": rating,
            "url": url,
            "review_count": len(comments),
            "comments": comments,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"Data extraction error {url}: {e}")
        return None

def main():
    global API_counter
    API_counter = 0
    json_path = "tum_restoranlar.json"
    restaurant_list = load_all_restaurant_links_with_names()
    print(f"📋 Toplam {len(restaurant_list)} restoran bulundu.")
    existing_data = read_existing_restaurants_json(json_path)
    # Devam edilecek restoranı bul
    start_idx = 0
    for idx, rest in enumerate(restaurant_list):
        if rest['link'] == "https://www.tripadvisor.com.tr/Restaurant_Review-g1221508-d25192216-Reviews-Kervan_Bistro-Kahramanmaras_Kahramanmaras_Province.html":
            start_idx = idx
            break
    for i, rest in enumerate(restaurant_list[start_idx:], start=start_idx+1):
        if API_counter >= API_LIMIT:
            print(f"API limiti ({API_LIMIT}) doldu. Durduruluyor.")
            break
        print(f"\n🔄 [{i}/{len(restaurant_list)}] Veri çekiliyor: {rest['name']} - {rest['link']}")
        restaurant_data = scrape_restaurant(rest['name'], rest['link'])
        if restaurant_data:
            # JSON'a ekle/güncelle
            existing_data = append_or_update_restaurant(existing_data, restaurant_data)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Kaydedildi: {restaurant_data['restaurant_name']} (Toplam yorum: {restaurant_data['review_count']})")
        else:
            print(f"❌ Başarısız: {rest['link']}")
        time.sleep(random.uniform(1, 2))
    print(f"\n🎉 İşlem tamamlandı veya API limiti doldu. Son kayıt: {json_path}")

if __name__ == "__main__":
    global API_counter
    API_counter = 0
    main()



