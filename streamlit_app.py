import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Kahramanmaraş Restoran Önerici",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .restaurant-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    .rating-stars {
        color: #ffc107;
        font-size: 1.2rem;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Veriyi yükle"""
    try:
        df = pd.read_csv('restaurants_with_reviews_sentiment.csv')
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None

@st.cache_resource
def load_model():
    """Modeli yükle"""
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return vectorizer
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

def recommend_restaurants(user_input, df, vectorizer, top_n=10):
    """Geliştirilmiş restoran önerisi"""
    
    if df is None or vectorizer is None:
        return pd.DataFrame()
    
    # Kullanıcı girdisini küçük harfe çevir
    user_input_lower = user_input.lower()
    
    # 1. Doğrudan restoran ismi eşleşmesi
    name_matches = df[df['restaurant_name'].str.lower().str.contains(user_input_lower, na=False, regex=False)]
    
    # 2. Food type eşleşmesi
    food_type_matches = df[df['food_types'].apply(lambda types: any(user_input_lower in str(food_type).lower() for food_type in types))]
    
    # 3. TF-IDF similarity
    try:
        # Combined text kullan (restoran ismi + yorumlar)
        combined_text = df['restaurant_name'] + ' ' + df['all_comments'].fillna('')
        user_vec = vectorizer.transform([user_input])
        X_tfidf = vectorizer.transform(combined_text)
        similarities = cosine_similarity(user_vec, X_tfidf).flatten()
        df_temp = df.copy()
        df_temp['similarity'] = similarities
        similarity_matches = df_temp[df_temp['similarity'] > 0.05]  # Daha düşük threshold
    except Exception as e:
        similarity_matches = pd.DataFrame()
    
    # Tüm eşleşmeleri birleştir
    all_matches = pd.concat([name_matches, food_type_matches, similarity_matches]).drop_duplicates(subset=['restaurant_name'])
    
    if all_matches.empty:
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
            food_types = row['food_types'] if isinstance(row['food_types'], list) else []
            food_type_bonus = 0.2 if any(user_input_lower in str(food_type).lower() for food_type in food_types) else 0.0
            
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
            return 0.1  # Varsayılan skor
    
    all_matches['final_score'] = all_matches.apply(calculate_final_score, axis=1)
    
    # NaN değerleri temizle
    all_matches = all_matches.dropna(subset=['final_score'])
    
    result = all_matches.sort_values(by='final_score', ascending=False).head(top_n)
    
    return result

def display_restaurant_card(restaurant, index):
    """Restoran kartını göster - güncellenmiş versiyon"""
    
    # Rating yıldızları
    rating = restaurant['rating']
    stars = '★' * int(rating) + '☆' * (5 - int(rating))
    
    # Skor yüzde formatında
    score_percentage = int(restaurant['final_score'] * 100)
    
    # Kart HTML'i - sadeleştirilmiş
    card_html = f"""
    <div class="restaurant-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">{index + 1}. {restaurant['restaurant_name']}</h3>
                <div style="margin-bottom: 0.5rem;">
                    <span class="rating-stars">{stars}</span>
                    <span style="color: #666; margin-left: 0.5rem;">{rating:.1f}/5.0</span>
                    <span style="color: #666; margin-left: 1rem;">💬 {restaurant['review_count']} yorum</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div class="score-badge">Öneri Skoru: %{score_percentage}</div>
                <br>
                <a href="{restaurant['url']}" target="_blank" style="text-decoration: none;">
                    <button style="background: #667eea; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer;">
                        🔗 Detaylar
                    </button>
                </a>
            </div>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def main():
    # Ana başlık
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ Kahramanmaraş Restoran Önerici</h1>
        <p>Yemek türü veya restoran adı yazarak size en uygun restoranları bulun</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Veriyi yükle
    df = load_data()
    vectorizer = load_model()
    
    if df is None or vectorizer is None:
        st.error("Veri veya model yüklenemedi. Lütfen dosyaların mevcut olduğundan emin olun.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Arama Seçenekleri")
        
        # Arama kutusu
        user_input = st.text_input(
            "Ne yemek istersiniz?",
            placeholder="Örn: kebap, dondurma, pasta, mado, kervan...",
            help="Yemek türü veya restoran adı yazın"
        )
        
        # Minimum rating filtresi
        min_rating = st.slider(
            "Minimum Puan:",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            help="Minimum restoran puanı"
        )
        
        # Maksimum sonuç sayısı
        max_results = st.slider(
            "Maksimum Sonuç Sayısı:",
            min_value=5,
            max_value=20,
            value=10,
            step=5
        )
        
        # Arama butonu
        search_button = st.button("🔍 Ara", type="primary", use_container_width=True)
    
    # Ana içerik
    if search_button and user_input.strip():
        with st.spinner("Aranıyor..."):
            # Restoran önerilerini al
            recommendations = recommend_restaurants(user_input, df, vectorizer, max_results)
            
            if recommendations.empty:
                st.warning(f"'{user_input}' için sonuç bulunamadı. Farklı bir arama terimi deneyin.")
            else:
                # Rating filtresi uygula
                recommendations = recommendations[recommendations['rating'] >= min_rating]
                
                if recommendations.empty:
                    st.warning("Seçilen kriterlere uygun restoran bulunamadı.")
                else:
                    st.success(f"'{user_input}' için {len(recommendations)} sonuç bulundu!")
                    
                    # Sonuçları göster
                    for idx, restaurant in recommendations.iterrows():
                        display_restaurant_card(restaurant, idx)
    
    elif not search_button:
        # Başlangıç ekranı
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
                <h3>🎯 Nasıl Kullanılır?</h3>
                <p>1. Sol taraftaki arama kutusuna yemek türü veya restoran adı yazın</p>
                <p>2. İsteğe bağlı olarak minimum puan ve maksimum sonuç sayısı seçin</p>
                <p>3. "Ara" butonuna tıklayın</p>
                <br>
                <h4>💡 Örnek Aramalar:</h4>
                <ul style="text-align: left; display: inline-block;">
                    <li>kebap - Kebap restoranları</li>
                    <li>dondurma - Dondurma mekanları</li>
                    <li>mado - Mado restoranı</li>
                    <li>kervan - Kervan Bistro</li>
                    <li>kahve - Cafe mekanları</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # İstatistikler - güncellenmiş
        if df is not None:
            st.subheader("📊 Veri İstatistikleri")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam Restoran", 175)
            
            with col2:
                st.metric("0 Yorumlu", 110)
            
            with col3:
                st.metric("Yorumlu", 65)
            
            with col4:
                st.metric("Ortalama Puan", f"{df['rating'].mean():.1f}")

if __name__ == "__main__":
    main() 