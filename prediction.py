import joblib
import pandas as pd

class RealEstatePredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    def predict(self, lat, lon, total_square, rooms, floor):
        # Создаем DataFrame с входными данными
        input_data = pd.DataFrame({
            'lat': [lat],
            'lon': [lon],
            'total_square': [total_square],
            'rooms': [rooms],
            'floor': [floor]
        })
        
        # Делаем предсказание
        prediction = self.model.predict(input_data)
        return prediction[0]

# Пример использования:
# predictor = RealEstatePredictor("real_estate_model.joblib")
# price = predictor.predict(55.75, 37.62, 100.0, 3, 10)
# print(f"Predicted price: {price:.2f} million rubles")