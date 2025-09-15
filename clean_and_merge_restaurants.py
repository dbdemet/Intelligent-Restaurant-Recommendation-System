import pandas as pd
import json

# JSON dosyasını oku
with open('restaurants_with_reviews.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Sadece restoran kayıtlarını filtrele (bazı kayıtlar eksik veya sadece scraped_at içerebilir)
records = [r for r in data if isinstance(r, dict) and 'restaurant_name' in r]

df = pd.DataFrame(records)

# all_comments sütununu oluştur
# Eğer comments sütunu yoksa veya boşsa, boş string ata

def merge_comments(comments):
    if isinstance(comments, list):
        return ' '.join(comments)
    return ''

df['all_comments'] = df['comments'].apply(merge_comments)

# Sadece istenen sütunlar
cols = ['restaurant_name', 'rating', 'review_count', 'url', 'all_comments']
df = df[cols]

# Null kontrolü ve doldurma
for col in cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna('')

# CSV olarak kaydet
csv_path = 'restaurants_with_reviews_cleaned.csv'
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"Temizlenmiş DataFrame CSV olarak kaydedildi: {csv_path}")
