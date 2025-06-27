import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def load_and_prepare_data(filepath):
    # Загрузка данных
    data = pd.read_csv(filepath)
    
    # Обработка пропущенных значений
    data['rooms'] = data['rooms'].fillna(data['rooms'].median())
    data['floor'] = data['floor'].fillna(data['floor'].median())
    
    # Выбор признаков и целевой переменной
    features = ['lat', 'lon', 'total_square', 'rooms', 'floor']
    target = 'price_mln'
    
    X = data[features]
    y = data[target]
    
    return X, y

def train_model(X, y):
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка модели
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae:.2f}")
    
    return model

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":
    # Загрузка и подготовка данных
    X, y = load_and_prepare_data("mylocaldata.csv")
    
    # Обучение модели
    model = train_model(X, y)
    
    # Сохранение модели
    save_model(model, "real_estate_model.joblib")
    print("Model trained and saved successfully.")