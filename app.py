import streamlit as st
from prediction import RealEstatePredictor

# Загрузка модели
@st.cache_resource
def load_model():
    return RealEstatePredictor("real_estate_model.joblib")

# Настройка страницы
st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏠")
st.title("🏠 Moscow Real Estate Price Predictor")
st.write("This app predicts the price of real estate in Moscow based on several features.")

# Создание формы для ввода данных
with st.form("prediction_form"):
    st.header("Enter Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lat = st.number_input("Latitude", min_value=55.0, max_value=56.0, value=55.75, step=0.01)
        total_square = st.number_input("Total Area (sq.m)", min_value=10.0, max_value=500.0, value=70.0, step=1.0)
        floor = st.number_input("Floor", min_value=1, max_value=50, value=5, step=1)
    
    with col2:
        lon = st.number_input("Longitude", min_value=37.0, max_value=38.5, value=37.62, step=0.01)
        rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=2, step=1)
    
    submitted = st.form_submit_button("Predict Price")

# Обработка отправки формы
if submitted:
    # Загрузка модели
    predictor = load_model()
    
    # Получение предсказания
    try:
        price = predictor.predict(lat, lon, total_square, rooms, floor)
        st.success(f"### Predicted Price: {price:.2f} million rubles")
        
        # Дополнительная визуализация
        st.write("### Price Breakdown")
        st.metric("Price per sq.m", f"{(price / total_square * 1000000):,.0f} rubles")
        
        # Карта с местоположением
        st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=10)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Дополнительная информация
st.sidebar.header("About")
st.sidebar.info(
    """
    This predictive model uses machine learning to estimate real estate prices in Moscow based on:
    - Location (latitude/longitude)
    - Total area
    - Number of rooms
    - Floor number
    
    The model was trained on historical transaction data.
    """
)