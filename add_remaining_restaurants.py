import json
import os

def load_source_restaurants():
    """Kaynak JSON dosyasından restoranları yükle"""
    source_file = "Restaurants-g1221508-Kahramanmaras_Kahramanmaras_Province/Restaurants-g1221508-Kahramanmaras_Kahramanmaras_Province/Restaurants-g1221508-Kahramanmaras_Kahramanmaras_Province_page_0.json"
    
    with open(source_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_existing_restaurants():
    """Mevcut tum_restoranlar.json dosyasını yükle"""
    if os.path.exists("tum_restoranlar.json"):
        with open("tum_restoranlar.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def find_kervan_bistro_index(restaurants):
    """Kervan Bistro'nun indeksini bul"""
    target_url = "https://www.tripadvisor.com.tr/Restaurant_Review-g1221508-d25192216-Reviews-Kervan_Bistro-Kahramanmaras_Kahramanmaras_Province.html"
    
    for i, restaurant in enumerate(restaurants):
        if restaurant.get('link') == target_url:
            return i
    return -1

def clean_restaurant_name(name):
    """Restoran adını temizle (numaraları kaldır)"""
    # Başındaki numarayı kaldır (örn: "77. Kervan Bistro" -> "Kervan Bistro")
    if '.' in name and name.split('.')[0].strip().isdigit():
        name = name[name.find('.')+1:].strip()
    return name

def create_restaurant_entry(restaurant):
    """Restoran verisi oluştur"""
    return {
        "restaurant_name": clean_restaurant_name(restaurant['name']),
        "rating": None,
        "url": restaurant['link'],
        "review_count": 0,
        "comments": [],
        "scraped_at": None
    }

def main():
    print("🚀 Kervan Bistro'dan itibaren restoranları ekleme işlemi başlıyor...")
    
    # Kaynak restoranları yükle
    source_restaurants = load_source_restaurants()
    print(f"📋 Kaynak dosyada {len(source_restaurants)} restoran bulundu")
    
    # Kervan Bistro'nun indeksini bul
    kervan_index = find_kervan_bistro_index(source_restaurants)
    if kervan_index == -1:
        print("❌ Kervan Bistro bulunamadı!")
        return
    
    print(f"🎯 Kervan Bistro {kervan_index + 1}. sırada bulundu")
    
    # Mevcut restoranları yükle
    existing_restaurants = load_existing_restaurants()
    print(f"📊 Mevcut dosyada {len(existing_restaurants)} restoran var")
    
    # Kervan Bistro'dan itibaren olan restoranları al
    remaining_restaurants = source_restaurants[kervan_index:]
    print(f"➕ {len(remaining_restaurants)} restoran eklenecek")
    
    # Mevcut URL'leri kontrol et (duplicate'leri önlemek için)
    existing_urls = {rest.get('url') for rest in existing_restaurants}
    
    # Yeni restoranları ekle
    added_count = 0
    for restaurant in remaining_restaurants:
        if restaurant['link'] not in existing_urls:
            new_entry = create_restaurant_entry(restaurant)
            existing_restaurants.append(new_entry)
            existing_urls.add(restaurant['link'])
            added_count += 1
            print(f"✅ {new_entry['restaurant_name']} eklendi")
        else:
            print(f"⏭️ {clean_restaurant_name(restaurant['name'])} zaten mevcut, atlanıyor")
    
    # Dosyayı kaydet
    with open("tum_restoranlar.json", 'w', encoding='utf-8') as f:
        json.dump(existing_restaurants, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 İşlem tamamlandı!")
    print(f"📊 Toplam {len(existing_restaurants)} restoran")
    print(f"➕ {added_count} yeni restoran eklendi")
    print(f"💾 Veriler tum_restoranlar.json dosyasına kaydedildi")

if __name__ == "__main__":
    main() 