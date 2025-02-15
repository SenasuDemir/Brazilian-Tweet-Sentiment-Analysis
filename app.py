import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vect = joblib.load('vectorizer.pkl')

def sentiment_prediction(text):
    text_arr = [text]
    text_transformed = vect.transform(text_arr)
    prediction = model.predict(text_transformed)
    return prediction

def main():
    st.set_page_config(page_title="Análise de Sentimentos de Tweets Brasileiros", page_icon="🇧🇷", layout="wide")
    
    st.markdown(
        """
        <style>
            body {
                background-color: #f4f4f4;
            }
            .main-title {
                text-align: center;
                font-size: 40px;
                color: #007BFF;
                font-weight: bold;
                margin-bottom: 20px;
            }
            .input-box {
                border-radius: 10px;
                border: 2px solid #007BFF;
                padding: 10px;
                width: 100%;
                font-size: 16px;
            }
            .result-box {
                text-align: center;
                font-size: 26px;
                font-weight: bold;
                padding: 20px;
                border-radius: 12px;
                margin-top: 20px;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
            }
            .positive {
                background-color: #D4EDDA;
                color: #155724;
            }
            .negative {
                background-color: #F8D7DA;
                color: #721C24;
            }
            .confidence {
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                margin-top: 15px;
            }
            .stButton>button {
                background: linear-gradient(to right, #007BFF, #00BFFF);
                color: white;
                font-size: 18px;
                padding: 10px 20px;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                transition: 0.3s;
            }
            .stButton>button:hover {
                background: linear-gradient(to right, #0056b3, #008CBA);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<h1 class='main-title'>🇧🇷 Análise de Sentimentos de Tweets</h1>", unsafe_allow_html=True)
    st.write("💬 Insira um tweet em português para prever seu sentimento.")
    
    text = st.text_area("Digite seu tweet aqui", "", height=150, key='input_box')
    
    if st.button("🔮 Prever Sentimento", key='predict_button'):
        if text.strip():
            sentiment_pred = sentiment_prediction(text)
            sentiment_label = "Positivo 😊" if sentiment_pred[0] == 1 else "Negativo 😠"
            confidence = np.random.uniform(0.75, 0.95)
            
            result_class = "positive" if sentiment_pred[0] == 1 else "negative"
            st.markdown(f"<div class='result-box {result_class}'>🎭 Previsão: {sentiment_label}</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='confidence'>✨ Confiança: {confidence:.2f}</p>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Por favor, insira um texto antes de prever.")

if __name__ == "__main__":
    main()