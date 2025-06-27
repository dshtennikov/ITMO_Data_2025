import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_and_save_model():
    # Загрузка данных (предполагается, что файл уже скачан)
    data = pd.read_csv("mylocaldata.csv")
    
    # Обработка данных
    data = data.dropna(subset=['lat', 'lon', 'total_square', 'price_mln'])
    data['rooms'] = data['rooms'].fillna(data['rooms'].median())
    data['floor'] = data['floor'].fillna(data['floor'].median())
    
    # Выбор признаков
    features = ['lat', 'lon', 'total_square', 'rooms', 'floor']
    X = data[features]
    y = data['price_mln']
    
    # Обучение модели
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Сохранение модели
    joblib.dump(model, "real_estate_model.joblib")
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_and_save_model()