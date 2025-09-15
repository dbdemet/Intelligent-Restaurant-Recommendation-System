from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import sys

app = Flask(__name__)

# Modeli ve veriyi yükle
def load_model():
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return vectorizer
    except:
        return None

def load_data():
    try:
        df = pd.read_csv('restaurants_with_reviews_sentiment.csv')
        return df
    except:
        return None

# Global değişkenler
vectorizer = load_model()
df = load_data()

def recommend_restaurants(user_input, top_n=5):
    """Kullanıcı girdisine göre restoran önerisi yap"""
    
    if df is None or vectorizer is None:
        return []
    
    # Kullanıcı girdisini küçük harfe çevir
    user_input_lower = user_input.lower()
    
    # 1. Doğrudan restoran ismi eşleşmesi - regex=False ekle
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
        similarity_matches = df_temp[df_temp['similarity'] > 0.1]  # Minimum similarity threshold
    except Exception as e:
        similarity_matches = pd.DataFrame()
    
    # Tüm eşleşmeleri birleştir
    all_matches = pd.concat([name_matches, food_type_matches, similarity_matches]).drop_duplicates(subset=['restaurant_name'])
    
    if all_matches.empty:
        return []
    
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
    
    # Sonuçları JSON formatına çevir
    recommendations = []
    for idx, row in result.iterrows():
        recommendations.append({
            'restaurant_name': row['restaurant_name'],
            'rating': float(row['rating']),
            'review_count': int(row['review_count']),
            'final_score': float(row['final_score']),
            'food_types': row['food_types'],
            'url': row['url']
        })
    
    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'recommendations': []})
    
    recommendations = recommend_restaurants(query, top_n=5)
    return jsonify({'recommendations': recommendations})

@app.route('/categories')
def get_categories():
    if df is None:
        return jsonify({'categories': []})
    
    # Tüm kategorileri topla
    all_categories = []
    for categories in df['food_types']:
        if isinstance(categories, list):
            all_categories.extend(categories)
        else:
            all_categories.append(categories)
    
    # Benzersiz kategorileri al ve sayıları hesapla
    category_counts = {}
    for category in all_categories:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Kategorileri sayıya göre sırala
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    return jsonify({'categories': [{'name': cat, 'count': count} for cat, count in sorted_categories]})

if __name__ == '__main__':
    # Windows'ta signal handling sorununu çözmek için
    if os.name == 'nt':  # Windows
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:  # Linux/Mac
        app.run(debug=True, host='0.0.0.0', port=5000)
