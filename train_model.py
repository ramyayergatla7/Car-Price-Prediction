import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train(df):

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Keep only required columns
    df = df[["brand", "year", "km_driven", "fuel", "price"]].dropna()

    # ----------------------------
    # Create Label Encoders
    # ----------------------------
    brand_encoder = LabelEncoder()
    fuel_encoder = LabelEncoder()

    df["brand"] = brand_encoder.fit_transform(df["brand"])
    df["fuel"] = fuel_encoder.fit_transform(df["fuel"])

    # Save encoders
    with open("brand_encoder.pkl", "wb") as f:
        pickle.dump(brand_encoder, f)

    with open("fuel_encoder.pkl", "wb") as f:
        pickle.dump(fuel_encoder, f)

    # ----------------------------
    # Features & Target
    # ----------------------------
    X = df[["brand", "year", "km_driven", "fuel"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ----------------------------
    # Models
    # ----------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    }

    scores = {}

    # ----------------------------
    # Train & Save Models
    # ----------------------------
    for name, model in models.items():

        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        scores[name] = score

        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    # ----------------------------
    # Best Model
    # ----------------------------
    best_model = max(scores, key=scores.get)

    # Save best model name
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return scores, best_model


# -------------------------------------------------
# Run directly from terminal (Optional)
# -------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv("used_car.csv")

    scores, best = train(df)

    print("\nModel Performance")
    print("-" * 30)

    for model, score in scores.items():
        print(f"{model}: {score:.4f}")

    print("\nBest Model:", best)