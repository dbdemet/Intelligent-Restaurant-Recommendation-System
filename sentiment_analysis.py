import pandas as pd
import json
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import joblib
import re

def load_all_restaurants():
    """restaurants_with_reviews.json dosyasından tüm restoranları yükle (0 yorumlu/rating'li olanlar dahil)"""
    
    with open('restaurants_with_reviews.json', 'r', encoding='utf-8') as f:
        restaurants = json.load(f)
    
    # DataFrame'e dönüştür
    data = []
    for restaurant in restaurants:
        # Yorumları birleştir
        comments = restaurant.get('comments', [])
        all_comments = ' '.join(comments) if comments else ''
        
        # Rating'i sayıya çevir
        rating_str = restaurant.get('rating', '0,0')
        if isinstance(rating_str, str):
            rating = float(rating_str.replace(',', '.'))
        else:
            rating = 0.0
        
        data.append({
            'restaurant_name': restaurant['restaurant_name'],
            'rating': rating,
            'url': restaurant['url'],
            'review_count': restaurant.get('review_count', 0),
            'all_comments': all_comments,
            'comments': comments
        })
    
    return pd.DataFrame(data)

# Tüm restoranları yükle
print("Restoranlar yükleniyor...")
df = load_all_restaurants()
print(f"Toplam {len(df)} restoran yüklendi")

# HuggingFace Türkçe duygu analizi pipeline'ı yükle
print("Sentiment analizi modeli yükleniyor...")
sentiment_pipeline = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0.5  # Varsayılan nötr skor
    try:
        result = sentiment_pipeline(text[:512])  # BERT max token limiti
        label = result[0]['label']
        score = result[0]['score']
        return score if label == "POSITIVE" else 1 - score
    except:
        return 0.5  # Hata durumunda nötr skor

# Sentiment skorunu hesapla (sadece yorumu olan restoranlar için)
print("Sentiment analizi başlatılıyor...")
df['sentiment_score'] = df['all_comments'].apply(get_sentiment)

def detect_price_score(text):
    if not isinstance(text, str):
        return 0.5
    text = text.lower()
    if "ucuz" in text or "makul" in text or "fiyatlar uygun" in text:
        return 1.0
    elif "pahalı" in text or "fiyat yüksek" in text:
        return 0.0
    else:
        return 0.5

# Fiyat algısı skorunu hesapla
print("Fiyat algısı analizi başlatılıyor...")
df['price_score'] = df['all_comments'].apply(detect_price_score)

# TF-IDF vektörleştirme öncesi null değerleri boş string ile doldur
if df['all_comments'].isnull().any():
    df['all_comments'] = df['all_comments'].fillna('')

# Restoran isimlerini de dahil etmek için combined_text oluştur
df['combined_text'] = df['restaurant_name'] + ' ' + df['all_comments']

# TF-IDF vektörleştirme (restoran isimleri + yorumlar)
print("TF-IDF vektörleştirme başlatılıyor...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df['combined_text'])

# Vektör ve vectorizer'ı kaydet
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('tfidf_vectors.npz', 'wb') as f:
    # scipy sparse matrix'i .npz olarak kaydet
    from scipy import sparse
    sparse.save_npz(f, X_tfidf)
print("TF-IDF vektörleri ve vectorizer kaydedildi.")

# Genişletilmiş yiyecek türü anahtar kelimeleri
food_keywords = {
    "kebap": ["kebap", "adana", "urfa", "şiş", "ciğer", "kuzu", "döner", "iskender", "paşa", "yoğurtlu", "kebapçı", "kebapçı"],
    "ızgara": ["köfte", "bonfile", "kuzu", "biftek", "pirzola", "antrikot", "tavuk ızgara", "ızgara", "ızgara"],
    "tavuk": ["tavuk", "kanat", "tavuk ızgara", "tavuk döner", "tavuk şiş", "tavukçı"],
    "balık": ["balık", "levrek", "çupra", "hamsi", "alabalık", "deniz ürünleri", "balıkçı"],
    "lahmacun_pide": ["lahmacun", "pide", "kaşarlı", "kıymalı", "kuşbaşılı", "peynirli", "lahmacuncu", "pideci"],
    "tatlı": ["tatlı", "baklava", "kadayıf", "künefe", "sütlaç", "muhallebi", "kazandibi", "pasta", "cheesecake", "tart", "profiterol", "ekler", "trileçe", "tiramisu", "şekerpare", "revani", "lokma", "kurabiye", "tatlıcı"],
    "dondurma": ["dondurma", "maras dondurması", "dondurmacı", "ice cream", "dondurma"],
    "çorba": ["mercimek", "ezogelin", "kelle paça", "paça", "işkembe", "çorba", "çorbacı"],
    "kahvaltı": ["kahvaltı", "serpme", "menemen", "yumurta", "omlet", "kahvaltıcı"],
    "fast_food": ["burger", "hamburger", "pizza", "nugget", "sandviç", "tost", "mcdonalds", "burger king", "fast food"],
    "salata": ["salata", "sebze", "humus", "meze", "salata"],
    "makarna": ["makarna", "spagetti", "fettucine", "penne", "tagliatelle", "makarna"],
    "ev_yemeği": ["kuru fasulye", "pilav", "musakka", "sulu yemek", "ev yemeği", "mercimek"],
    "cafe": ["cafe", "kahve", "latte", "americano", "espresso", "çay", "kafe", "kahve", "coffee"],
    "pastane": ["pastane", "pasta", "kurabiye", "börek", "poğaça", "çörek", "ekler", "tart", "cheesecake", "pastacı"],
    "et_lokantası": ["et", "lokanta", "et lokantası", "kasap", "etçi", "et lokantası"],
    "çiğ_köfte": ["çiğ köfte", "çiğköfte", "çiğköfteci"],
    "tantuni": ["tantuni", "tantun", "tantunicı"],
    "pide": ["pide", "kaşarlı pide", "kıymalı pide", "pideci"],
    "lahmacun": ["lahmacun", "lahmacuncu", "lahmacun"]
}

def extract_food_types(restaurant_name, comments_text):
    """Restoran ismine ve yorumlara bakarak yemek türlerini belirle"""
    text = (str(restaurant_name) + ' ' + str(comments_text)).lower()
    detected = []
    
    for food_type, keywords in food_keywords.items():
        if any(keyword in text for keyword in keywords):
            detected.append(food_type)
    
    # Restoran ismine özel kontroller
    restaurant_name_lower = str(restaurant_name).lower()
    
    # Özel restoranlar için kategoriler
    special_restaurants = {
        'tatlıpark': ['tatlı', 'dondurma', 'cafe'],
        'mado': ['dondurma', 'tatlı', 'cafe'],
        'akdo': ['dondurma', 'tatlı', 'cafe'],
        'kervan': ['cafe', 'tatlı', 'dondurma'],
        'starbucks': ['cafe'],
        'gloria jeans': ['cafe'],
        'dunkin': ['cafe', 'tatlı'],
        'simit sarayı': ['pastane', 'kahvaltı'],
        'börekçi': ['pastane'],
        'poğaçacı': ['pastane'],
        'çörekçi': ['pastane'],
        'kurabiyeci': ['pastane'],
        'ekmekçi': ['pastane'],
        'fırın': ['pastane'],
        'pizzacı': ['fast_food'],
        'burgerci': ['fast_food'],
        'mcdonalds': ['fast_food'],
        'burger king': ['fast_food'],
        'kfc': ['fast_food'],
        'subway': ['fast_food'],
        'dominos': ['fast_food'],
        'pizza hut': ['fast_food'],
        'kebapçı': ['kebap'],
        'adana kebap': ['kebap'],
        'urfa kebap': ['kebap'],
        'ciğerci': ['kebap'],
        'dönerci': ['kebap'],
        'iskender': ['kebap'],
        'lahmacuncu': ['lahmacun_pide'],
        'pideci': ['pide'],
        'çiğköfteci': ['çiğ_köfte'],
        'tantunicı': ['tantuni'],
        'balıkçı': ['balık'],
        'tavukçı': ['tavuk'],
        'köfteci': ['ızgara'],
        'ızgara': ['ızgara'],
        'çorbacı': ['çorba'],
        'kahvaltıcı': ['kahvaltı'],
        'tatlıcı': ['tatlı'],
        'dondurmacı': ['dondurma'],
        'pastacı': ['pastane'],
        'etçi': ['et_lokantası'],
        'kasap': ['et_lokantası'],
        'lokanta': ['et_lokantası'],
        'restoran': ['et_lokantası'],
        'salata': ['salata'],
        'makarna': ['makarna'],
        'ev yemeği': ['ev_yemeği'],
        'sulu yemek': ['ev_yemeği']
    }
    
    # Özel restoran kontrolü
    for keyword, categories in special_restaurants.items():
        if keyword in restaurant_name_lower:
            detected.extend(categories)
    
    # Genel kategori kontrolleri
    if any(word in restaurant_name_lower for word in ['pastane', 'pastacı', 'tatlı']):
        detected.append('pastane')
    if any(word in restaurant_name_lower for word in ['cafe', 'kafe', 'kahve']):
        detected.append('cafe')
    if any(word in restaurant_name_lower for word in ['et', 'kasap', 'kebap']):
        detected.append('et_lokantası')
    if any(word in restaurant_name_lower for word in ['pizza', 'pizzacı']):
        detected.append('fast_food')
    if any(word in restaurant_name_lower for word in ['burger', 'mcdonalds']):
        detected.append('fast_food')
    if any(word in restaurant_name_lower for word in ['dondurma', 'dondurmacı']):
        detected.append('dondurma')
    if any(word in restaurant_name_lower for word in ['çiğ köfte', 'çiğköfte']):
        detected.append('çiğ_köfte')
    if any(word in restaurant_name_lower for word in ['tantuni', 'tantun']):
        detected.append('tantuni')
    if any(word in restaurant_name_lower for word in ['lahmacun', 'lahmacuncu']):
        detected.append('lahmacun_pide')
    if any(word in restaurant_name_lower for word in ['pide', 'pideci']):
        detected.append('pide')
    if any(word in restaurant_name_lower for word in ['balık', 'balıkçı']):
        detected.append('balık')
    if any(word in restaurant_name_lower for word in ['tavuk', 'tavukçı']):
        detected.append('tavuk')
    if any(word in restaurant_name_lower for word in ['köfte', 'köfteci']):
        detected.append('ızgara')
    if any(word in restaurant_name_lower for word in ['ızgara']):
        detected.append('ızgara')
    if any(word in restaurant_name_lower for word in ['çorba', 'çorbacı']):
        detected.append('çorba')
    if any(word in restaurant_name_lower for word in ['kahvaltı', 'kahvaltıcı']):
        detected.append('kahvaltı')
    if any(word in restaurant_name_lower for word in ['tatlı', 'tatlıcı']):
        detected.append('tatlı')
    if any(word in restaurant_name_lower for word in ['salata']):
        detected.append('salata')
    if any(word in restaurant_name_lower for word in ['makarna']):
        detected.append('makarna')
    if any(word in restaurant_name_lower for word in ['ev yemeği', 'sulu yemek']):
        detected.append('ev_yemeği')
    
    return list(set(detected)) if detected else ["diğer"]

# Yemek türlerini belirle
print("Yemek türleri belirleniyor...")
df['food_types'] = df.apply(lambda row: extract_food_types(row['restaurant_name'], row['all_comments']), axis=1)

# Geliştirilmiş restoran öneri fonksiyonu
def recommend_restaurants(user_input):
    """Geliştirilmiş arama algoritması - restoran isimleri ve kategorileri de dahil"""
    
    # Kullanıcı girdisini küçük harfe çevir
    user_input_lower = user_input.lower()
    
    # 1. Doğrudan restoran ismi eşleşmesi
    name_matches = df[df['restaurant_name'].str.lower().str.contains(user_input_lower, na=False)]
    
    # 2. Food type eşleşmesi
    food_type_matches = df[df['food_types'].apply(lambda types: any(user_input_lower in food_type for food_type in types))]
    
    # 3. TF-IDF similarity
    try:
        user_vec = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vec, X_tfidf).flatten()
        df_temp = df.copy()
        df_temp['similarity'] = similarities
        similarity_matches = df_temp[df_temp['similarity'] > 0.1]  # Minimum similarity threshold
    except:
        similarity_matches = pd.DataFrame()
    
    # Tüm eşleşmeleri birleştir
    all_matches = pd.concat([name_matches, food_type_matches, similarity_matches]).drop_duplicates(subset=['restaurant_name'])
    
    if all_matches.empty:
        return pd.DataFrame()
    
    # Final skor hesaplama
    def calculate_final_score(row):
        # Temel skorlar
        sentiment = row['sentiment_score'] if row['review_count'] > 0 else 0.5
        price = row['price_score']
        similarity = row.get('similarity', 0.1)
        
        # Restoran ismi eşleşmesi bonusu
        name_bonus = 0.3 if user_input_lower in row['restaurant_name'].lower() else 0.0
        
        # Food type eşleşmesi bonusu
        food_type_bonus = 0.2 if any(user_input_lower in food_type for food_type in row['food_types']) else 0.0
        
        # Yorum sayısına göre ağırlık
        review_weight = min(row['review_count'] / 10, 1.0) if row['review_count'] > 0 else 0.3
        
        # Final skor hesaplama
        if row['review_count'] > 0:
            final_score = (0.3 * sentiment + 0.2 * price + 0.3 * similarity + name_bonus + food_type_bonus) * review_weight
        else:
            # 0 yorumlu restoranlar için
            final_score = 0.4 * similarity + 0.2 * price + name_bonus + food_type_bonus
        
        return final_score
    
    all_matches['final_score'] = all_matches.apply(calculate_final_score, axis=1)
    return all_matches.sort_values(by='final_score', ascending=False).head(10)

# Yemek türüne göre öneri fonksiyonu
def recommend_by_food_type(food_type, user_input):
    """Belirli bir yemek türüne göre öneri"""
    # Seçilen yemek türüne ait restoranlar filtrelenir
    filtered_df = df[df['food_types'].apply(lambda types: food_type in types)]
    if filtered_df.empty:
        return pd.DataFrame()
    
    try:
        user_vec = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vec, vectorizer.transform(filtered_df['combined_text'])).flatten()
        filtered_df = filtered_df.copy()
        filtered_df['similarity'] = similarities
    except:
        filtered_df['similarity'] = 0.1
    
    # Final skor hesaplama
    def weight_score(row):
        if row['review_count'] > 0:
            score = 0.4 * row['sentiment_score'] + 0.2 * row['price_score'] + 0.4 * row['similarity']
            # Yorum sayısına göre ağırlık
            review_weight = min(row['review_count'] / 10, 1.0)
            return score * review_weight
        else:
            # 0 yorumlu restoranlar için
            return 0.6 * row['similarity'] + 0.4 * row['price_score']
    
    filtered_df['final_score'] = filtered_df.apply(weight_score, axis=1)
    return filtered_df.sort_values(by='final_score', ascending=False).head(10)

# Model ve DataFrame'i kaydet
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
df.to_pickle("restaurant_data_with_scores.pkl")

# Sonuçları yeni CSV'ye kaydet
out_path = 'restaurants_with_reviews_sentiment.csv'
df.to_csv(out_path, index=False, encoding='utf-8')
print(f"Sentiment ve fiyat algısı skorları eklendi ve kaydedildi: {out_path}")

# Her food_type için ayrı TF-IDF vectorizer eğit ve kaydet
from collections import defaultdict
vectorizers = {}
food_types_all = sorted(set(ft for sublist in df['food_types'] for ft in sublist))
for ftype in food_types_all:
    sub_df = df[df['food_types'].apply(lambda types: ftype in types)]
    # En az 1 anlamlı yorum olmalı ve boş olmamalı
    comments = [str(c).strip() for c in sub_df['combined_text'] if isinstance(c, str) and str(c).strip()]
    if len(comments) > 0:
        vec = TfidfVectorizer(stop_words=None, max_features=500)
        try:
            vec.fit(comments)
            vectorizers[ftype] = vec
            joblib.dump(vec, f"vectorizer_{ftype}.pkl")
        except ValueError:
            pass  # Boş vocabulary hatası varsa atla

# Örnek kullanım
if __name__ == "__main__":
    print("Öneri örneği: kebap için en iyi restoranlar")
    recommendations = recommend_restaurants("kebap")
    print(recommendations[["restaurant_name", "final_score", "sentiment_score", "price_score", "similarity", "review_count", "food_types"]].head())
    
    print("\nYemek türüne göre öneri örneği: kebap türü için")
    food_recommendations = recommend_by_food_type("kebap", "kebap")
    print(food_recommendations[["restaurant_name", "final_score", "sentiment_score", "price_score", "similarity", "review_count", "food_types"]].head())
