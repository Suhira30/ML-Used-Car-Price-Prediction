import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
from datetime import date
import xgboost as xgb

MODEL_PATH = "car_price_model.pkl"
COLS_PATH  = "car_price_columns.pkl"

def get_ohe_categories(pipeline):
    preprocessor = pipeline.named_steps["preprocess"]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    
    cat_cols = preprocessor.transformers_[1][2]  # categorical column list
    
    categories = {}
    for col, cats in zip(cat_cols, ohe.categories_):
        categories[col] = list(cats)
    
    return categories

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(COLS_PATH, "rb") as f:
        cols = pickle.load(f)
    return model, cols

model, cols = load_artifacts()
cat_options = get_ohe_categories(model)

def build_shap_tools(pipeline, cols):
    preprocessor = pipeline.named_steps["preprocess"]
    xgb_model = pipeline.named_steps["model"]
    booster = xgb_model.get_booster()

    cat_cols = [c for c in cols if c in [
        "Brand", "Model", "Gear", "Fuel Type", "Town", "Leasing", "Condition",
        "AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"
    ]]
    num_cols = [c for c in cols if c not in cat_cols]

    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    feature_names = np.concatenate([num_cols, cat_feature_names])

    return booster, preprocessor, feature_names

def explain_with_shap(pipeline, cols, X_row_df):
    booster, preprocessor, feature_names = build_shap_tools(pipeline, cols)

    X_t = preprocessor.transform(X_row_df)

    pred = float(pipeline.predict(X_row_df)[0])

    dmat = xgb.DMatrix(X_t, feature_names=list(feature_names))
    contrib = booster.predict(dmat, pred_contribs=True)[0]   # shape: (n_features + 1,)

    base_val = float(contrib[-1])
    sv_1d = np.array(contrib[:-1], dtype=float)

    if hasattr(X_t, "toarray"):
        x_1d = X_t.toarray().ravel()
    else:
        x_1d = np.array(X_t).ravel()

    df_shap = pd.DataFrame({
        "feature": feature_names,
        "value": x_1d,
        "shap_value": sv_1d,
        "abs_shap": np.abs(sv_1d)
    }).sort_values("abs_shap", ascending=False)

    top = df_shap.head(15).drop(columns=["abs_shap"]).reset_index(drop=True)

    exp = shap.Explanation(
        values=sv_1d,
        base_values=base_val,
        data=x_1d,
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(9, 6))
    try:
        shap.plots.waterfall(exp, max_display=15, show=False)
    except TypeError:
        shap.plots.waterfall(exp, max_display=15)
    plt.title("SHAP-style Waterfall (XGBoost pred_contribs)")
    plt.tight_layout()

    return pred, top, fig

st.title("Car Price Prediction (Sri Lanka)")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", cat_options["Brand"])
    model_name = st.selectbox("Model", cat_options["Model"])
    yom = st.number_input("YOM", min_value=1950, max_value=2030, value=2017, step=1)
    engine_cc = st.number_input("Engine (cc)", min_value=500.0, max_value=10000.0, value=1000.0, step=50.0)
    gear = st.selectbox("Gear", ["Automatic", "Manual"])

with col2:
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
    mileage = st.number_input("Millage(KM)", min_value=0.0, max_value=2000000.0, value=88000.0, step=1000.0)
    town = st.selectbox("Town", cat_options["Town"])
    leasing = st.selectbox("Leasing", ["No Leasing", "Leasing Available"])
    condition = st.selectbox("Condition", ["USED", "NEW", "RECONDITIONED"])

st.subheader("Features")
c1, c2, c3, c4 = st.columns(4)
with c1:
    air = st.selectbox("AIR CONDITION", ["Available", "Not_Available"])
with c2:
    ps = st.selectbox("POWER STEERING", ["Available", "Not_Available"])
with c3:
    pm = st.selectbox("POWER MIRROR", ["Available", "Not_Available"])
with c4:
    pw = st.selectbox("POWER WINDOW", ["Available", "Not_Available"])

st.subheader("Listing info")
today = date.today()
default_year = today.year
default_month = today.month

listing_year = st.number_input("ListingYear", min_value=2000, max_value=2035, value=int(default_year), step=1)
listing_month = st.number_input("ListingMonth", min_value=1, max_value=12, value=int(default_month), step=1)

TRAIN_CURRENT_YEAR = 2025
vehicle_age = TRAIN_CURRENT_YEAR - int(yom)

row = {
    "Brand": brand,
    "Model": model_name,
    "YOM": int(yom),
    "Engine (cc)": float(engine_cc),
    "Gear": gear,
    "Fuel Type": fuel,
    "Millage(KM)": float(mileage),
    "Town": town,
    "Leasing": leasing,
    "Condition": condition,
    "AIR CONDITION": air,
    "POWER STEERING": ps,
    "POWER MIRROR": pm,
    "POWER WINDOW": pw,
    "ListingYear": int(listing_year),
    "ListingMonth": int(listing_month),
    "VehicleAge": int(vehicle_age)
}

X_input = pd.DataFrame([row]).reindex(columns=cols)

do_predict = st.button("Predict Price")
do_explain = st.checkbox("Show SHAP explanation for this prediction")

if do_predict:
    pred = float(model.predict(X_input)[0])
    st.success(f"Predicted Price: {pred:.2f} lakhs")
    st.caption("Note: Prediction is based on historical listings and may not reflect negotiation/market changes.")

    if do_explain:
        with st.spinner("Computing SHAP explanation..."):
            pred2, top_df, fig = explain_with_shap(model, cols, X_input)

        st.subheader("Top SHAP feature contributions")
        st.dataframe(top_df, use_container_width=True)

        st.subheader("Waterfall plot")
        st.pyplot(fig, clear_figure=True)