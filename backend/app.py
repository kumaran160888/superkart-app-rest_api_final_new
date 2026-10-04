# Import necessary libraries
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

superkart_sales_application_api = Flask("superkart_sales_application")

# Load the trained pipeline (preprocessing + model)
model = joblib.load("superkart_sales_model.joblib")

# ----------------------------------------------------------------------
# The model was trained on these column names:
#   numeric:     Product_Weight, Product_Allocated_Area, Product_MRP, Store_Age
#   categorical: Product_Sugar_Content, Product_Type, Store_Size,
#                Store_Location_City_Type, Store_Type
#
# The incoming payload uses slightly different names. We translate here
# so the sklearn Pipeline sees exactly what it expects.
# ----------------------------------------------------------------------

# Field-name mapping: payload key -> model's training column name
FIELD_RENAME = {
    "Store_Age_Years": "Store_Age",
}

# Product_Type_Category (Perishables / Non Perishables) does NOT exist
# in the trained model. We map it to a representative Product_Type value
# the model DID see during training.
PRODUCT_TYPE_CATEGORY_MAP = {
    "Perishables":     "Dairy",
    "Non Perishables": "Snack Foods",
}

# Columns the model expects
EXPECTED_COLUMNS = []
    "Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age",
    "Product_Sugar_Content", "Product_Type", "Store_Size",
    "Store_Location_City_Type", "Store_Type",
]


def transform_payload(sample: dict) -> dict:
    """Translate the API payload into the model's training schema."""
    row = dict(sample)  # copy so we don't mutate the caller's dict

    # 1. Rename fields
    for old, new in FIELD_RENAME.items():
        if old in row:
            row[new] = row.pop(old)

    # 2. Map Product_Type_Category -> Product_Type
    if "Product_Type_Category" in row:
        category = row.pop("Product_Type_Category")
        row["Product_Type"] = PRODUCT_TYPE_CATEGORY_MAP.get(category, "Snack Foods")

    # 3. Drop any field the model wasn't trained on (e.g. Product_Id_char)
    row = {k: v for k, v in row.items() if k in EXPECTED_COLUMNS}

    return row


@superkart_sales_application_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


@superkart_sales_application_api.post('/v1/predict')
def predict_sales_price():
    try:
        sample = request.get_json()
        transformed = transform_payload(sample)
        input_data = pd.DataFrame([transformed])

        predicted_sales = model.predict(input_data)[0]
        predicted_sales = round(float(predicted_sales), 2)

        return jsonify({"Predicted_Sales": predicted_sales})
    except Exception as e:
        # Return the error as JSON so the client doesn't get an HTML 500 page
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@superkart_sales_application_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        file = request.files['file']
        batch_data = pd.read_csv(file)

        # Apply the same transformation to each row
        transformed_rows = [transform_payload(row) for row in batch_data.to_dict(orient="records")]
        transformed_df = pd.DataFrame(transformed_rows)

        predictions = model.predict(transformed_df)
        result = {str(i): round(float(p), 2) for i, p in enumerate(predictions)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


if __name__ == '__main__':
    superkart_sales_application_api.run(host='0.0.0.0', port=7860, debug=True)
