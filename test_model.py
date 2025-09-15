import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Modeli yükle
def load_model():
    try:
        # Önce joblib ile dene
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except:
        # Eğer joblib çalışmazsa pickle ile dene
        import pickle
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    return vectorizer

# Veriyi yükle
df = pd.read_csv('restaurants_with_reviews_sentiment.csv')

# Modeli yükle
vectorizer = load_model()

def recommend_restaurants(user_input, top_n=5):
    """Kullanıcı girdisine göre restoran önerisi yap"""
    
    # Kullanıcı girdisini küçük harfe çevir
    user_input_lower = user_input.lower()
    
    # 1. Doğrudan restoran ismi eşleşmesi
    name_matches = df[df['restaurant_name'].str.lower().str.contains(user_input_lower, na=False)]
    
    # 2. Food type eşleşmesi
    food_type_matches = df[df['food_types'].apply(lambda types: any(user_input_lower in food_type for food_type in types))]
    
    # 3. TF-IDF similarity
    try:
        # Combined text kullan (restoran ismi + yorumlar)
        combined_text = df['restaurant_name'] + ' ' + df['all_comments'].fillna('')
        user_vec = vectorizer.transform([user_input])
        X_tfidf = vectorizer.transform(combined_text)
        similarities = cosine_similarity(user_vec, X_tfidf).flatten()
        df_temp = df.copy()
        df_temp['similarity'] = similarities
        similarity_matches = df_temp[df_temp['similarity'] > 0.1]  # Minimum similarity threshold
    except Exception as e:
        print(f"Similarity hesaplama hatası: {e}")
        similarity_matches = pd.DataFrame()
    
    # Tüm eşleşmeleri birleştir
    all_matches = pd.concat([name_matches, food_type_matches, similarity_matches]).drop_duplicates(subset=['restaurant_name'])
    
    if all_matches.empty:
        print(f"'{user_input}' için hiç restoran bulunamadı.")
        return pd.DataFrame()
    
    # Final skor hesaplama
    def calculate_final_score(row):
        try:
            # Temel skorlar - güvenli şekilde al
            sentiment = float(row['sentiment_score']) if pd.notna(row['sentiment_score']) else 0.5
            price = float(row['price_score']) if pd.notna(row['price_score']) else 0.5
            similarity = float(row.get('similarity', 0.1)) if pd.notna(row.get('similarity', 0.1)) else 0.1
            
            # Restoran ismi eşleşmesi bonusu
            name_bonus = 0.3 if user_input_lower in row['restaurant_name'].lower() else 0.0
            
            # Food type eşleşmesi bonusu
            food_type_bonus = 0.2 if any(user_input_lower in food_type for food_type in row['food_types']) else 0.0
            
            # Yorum sayısına göre ağırlık
            review_count = int(row['review_count']) if pd.notna(row['review_count']) else 0
            review_weight = min(review_count / 10, 1.0) if review_count > 0 else 0.3
            
            # Final skor hesaplama
            if review_count > 0:
                final_score = (0.3 * sentiment + 0.2 * price + 0.3 * similarity + name_bonus + food_type_bonus) * review_weight
            else:
                # 0 yorumlu restoranlar için
                final_score = 0.4 * similarity + 0.2 * price + name_bonus + food_type_bonus
            
            return final_score
        except Exception as e:
            print(f"Skor hesaplama hatası: {e}")
            return 0.1  # Varsayılan skor
    
    all_matches['final_score'] = all_matches.apply(calculate_final_score, axis=1)
    
    # NaN değerleri temizle
    all_matches = all_matches.dropna(subset=['final_score'])
    
    result = all_matches.sort_values(by='final_score', ascending=False).head(top_n)
    
    return result[['restaurant_name', 'final_score', 'rating', 'review_count', 'food_types']]

# Test fonksiyonu
def test_searches():
    """Farklı arama terimlerini test et"""
    
    test_terms = [
        "kebap",
        "dondurma", 
        "pasta",
        "makarna",
        "çiğköfte",
        "hamburger",
        "kahve",
        "tatlıpark",
        "mado",
        "akdo",
        "kervan"
    ]
    
    print("🔍 Arama Testleri Başlıyor...\n")
    
    for term in test_terms:
        print(f"📝 Arama terimi: '{term}'")
        results = recommend_restaurants(term, top_n=3)
        
        if not results.empty:
            print(f"✅ {len(results)} sonuç bulundu:")
            for idx, row in results.iterrows():
                print(f"   🏪 {row['restaurant_name']} (Skor: {row['final_score']:.3f}, Rating: {row['rating']}, Yorum: {row['review_count']})")
                print(f"      Kategoriler: {row['food_types']}")
        else:
            print("❌ Sonuç bulunamadı")
        
        print("-" * 50)

if __name__ == "__main__":
    test_searches() 