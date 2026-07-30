import streamlit as st
import pandas as pd
import pickle
import os
from train_model import train

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Prediction")

uploaded_file = st.file_uploader(
    "Upload Used Car Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()

    required_columns = ["brand", "year", "km_driven", "fuel", "price"]

    if not all(col in df.columns for col in required_columns):
        st.error("Dataset must contain: brand, year, km_driven, fuel and price")
        st.stop()

    st.success("Dataset uploaded successfully!")

    # Train models only if they don't exist
    if (
        not os.path.exists("Linear Regression.pkl")
        or not os.path.exists("Decision Tree.pkl")
        or not os.path.exists("Random Forest.pkl")
        or not os.path.exists("brand_encoder.pkl")
        or not os.path.exists("fuel_encoder.pkl")
    ):
        with st.spinner("Training models..."):
            scores, best_model = train(df)
    else:
        # Train once to determine best model (optional)
        scores, best_model = train(df)

    st.subheader("Model Performance")

    for model, score in scores.items():
        st.write(f"**{model}** : {score:.4f}")

    st.success(f"Best Model : {best_model}")

    # Load trained model
    model = pickle.load(open(best_model + ".pkl", "rb"))

    # Load encoders
    brand_encoder = pickle.load(open("brand_encoder.pkl", "rb"))
    fuel_encoder = pickle.load(open("fuel_encoder.pkl", "rb"))
    
    brand = st.selectbox(
        "Brand",
        sorted(brand_encoder.classes_)
    )
    year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=2035,
        value=2020
    )
    km_driven = st.number_input(
        "KM Driven",
        min_value=0,
        value=50000,
        step=1000
    )
    fuel = st.selectbox(
        "Fuel Type",
        sorted(fuel_encoder.classes_)
    )
    if st.button("Predict Price"):
        try:
            brand_value = brand_encoder.transform([brand])[0]
            fuel_value = fuel_encoder.transform([fuel])[0]

            prediction = model.predict([[
                brand_value,
                year,
                km_driven,
                fuel_value
            ]])[0]
            st.success(
                f"💰 Estimated Price: ₹{prediction:,.0f}"
            )
        except ValueError:
            st.error("Unknown Brand or Fuel Type.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")