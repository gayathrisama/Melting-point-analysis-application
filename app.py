import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.title("Melting Point Analysis App")

# Introduction
st.subheader("Introduction")
st.write("This app allows users to upload a melting point dataset and explore it using filtering, visualization, and machine learning.")

uploaded_file = st.file_uploader("Upload your melting point dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Show columns
    st.subheader("Column Info")
    st.write(df.columns.tolist())

    # Filter section
    st.subheader("Filter Dataset")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    selected_col = st.selectbox("Select a column to filter", numeric_cols)

    min_val, max_val = st.slider(
        "Select range",
        float(df[selected_col].min()),
        float(df[selected_col].max()),
        (float(df[selected_col].min()), float(df[selected_col].max()))
    )

    filtered_df = df[(df[selected_col] >= min_val) & (df[selected_col] <= max_val)]
    st.write(filtered_df)

    # Visualization
    st.subheader("Visualization: Histogram")
    fig, ax = plt.subplots()
    ax.hist(df[selected_col].dropna())
    st.pyplot(fig)

    # Machine Learning Model
    st.subheader("Train Model (Random Forest)")
    target = st.selectbox("Select target column (melting point)", numeric_cols)

    feature_cols = st.multiselect("Select feature columns", numeric_cols)

    if len(feature_cols) > 0:
        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        st.write("### Model MAE:", mae)

        # Predict new value
        st.subheader("Predict Melting Point")
        input_values = []

        for col in feature_cols:
            val = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
            input_values.append(val)

        if st.button("Predict"):
            pred = model.predict([input_values])[0]
            st.success(f"Predicted Melting Point: {pred}")

# Conclusion
st.subheader("Conclusion")
st.write("This project demonstrates data uploading, filtering, visualization, and machine learning using Streamlit.")

