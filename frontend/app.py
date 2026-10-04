import streamlit as st
import requests
import pandas as pd
import io

st.title("🛒 SuperKart Sales Forecasting App")

BACKEND_URL = "http://localhost:7860"

# =====================================================================
# TAB LAYOUT: Single prediction (form) + Batch prediction (CSV upload)
# =====================================================================
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction (CSV)"])

# ---------------------------------------------------------------------
# TAB 1 — single prediction via form
# ---------------------------------------------------------------------
with tab1:
    st.markdown("🔍 Enter product and store attributes to forecast **monthly product sales revenue**.\n\n_All sales are reported in ($) USD._")

    Product_Weight = st.number_input("Product Weight (oz)", min_value=0.0, value=12.66)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Allocated_Area = st.number_input("Product Allocated Area (linear in.)", min_value=0.0, value=100.0)
    Product_MRP = st.number_input("Maximum Retail Price (USD)", min_value=0.0, value=150.0)
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    Store_Age_Years = st.slider("Store Age (years)", min_value=0, max_value=30, value=10)
    Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    product_data = {
        "Product_Weight": Product_Weight,
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_MRP": Product_MRP,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type,
        "Store_Age_Years": Store_Age_Years,
        "Product_Type_Category": Product_Type_Category,
    }

    if st.button("Predict", type='primary', key="single_predict"):
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=15)
            if response.status_code == 200:
                result = response.json()
                st.success(f"📈 Predicted Monthly Sales: **${result['Predicted_Sales']:,.2f} USD**")
            else:
                try:
                    err = response.json()
                    st.error(f"❌ API Error ({response.status_code}): {err.get('error', 'unknown')}")
                except Exception:
                    st.error(f"❌ API Error ({response.status_code}): {response.text[:200]}")
        except Exception as e:
            st.error(f"⚠️ Connection error: {e}")


# ---------------------------------------------------------------------
# TAB 2 — batch prediction via CSV upload
# ---------------------------------------------------------------------
with tab2:
    st.markdown("""
    📂 Upload a CSV file to predict sales for multiple products at once.

    **Required columns:**
    `Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, `Product_MRP`,
    `Store_Size`, `Store_Location_City_Type`, `Store_Type`, `Store_Age_Years`, `Product_Type_Category`
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Preview the uploaded CSV
            df_preview = pd.read_csv(uploaded_file)
            st.write(f"✅ File loaded: **{uploaded_file.name}** ({len(df_preview)} rows)")
            st.dataframe(df_preview.head(10), use_container_width=True)

            # Reset the file pointer so we can send it to the API
            uploaded_file.seek(0)

            if st.button("Predict Batch", type='primary', key="batch_predict"):
                with st.spinner("Running batch prediction…"):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        response = requests.post(
                            f"{BACKEND_URL}/v1/predictbatch",
                            files=files,
                            timeout=60
                        )

                        if response.status_code == 200:
                            predictions = response.json()
                            # predictions is {"0": 123.45, "1": 234.56, ...}
                            pred_series = pd.Series(predictions).sort_index()
                            pred_series.index = pred_series.index.astype(int)

                            result_df = df_preview.copy()
                            result_df["Predicted_Sales (USD)"] = pred_series.values

                            st.success(f"📊 Predicted sales for {len(result_df)} rows")
                            st.dataframe(result_df, use_container_width=True)

                            # Offer download of predictions
                            csv_out = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "⬇️ Download predictions as CSV",
                                data=csv_out,
                                file_name="superkart_predictions.csv",
                                mime="text/csv"
                            )
                        else:
                            try:
                                err = response.json()
                                st.error(f"❌ API Error ({response.status_code}): {err.get('error', 'unknown')}")
                            except Exception:
                                st.error(f"❌ API Error ({response.status_code}): {response.text[:500]}")
                    except Exception as e:
                        st.error(f"⚠️ Connection error: {e}")

        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
