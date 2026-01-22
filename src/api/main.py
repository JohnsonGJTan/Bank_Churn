import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from src.api.schema import ChurnPredictionInput
import io
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from src.visualizations.shap_utils import get_shap_values, make_beeswarm_viz

api = FastAPI(title= 'Bank Churn Prediction API')

MODEL_PATH = "pipelines/final_model.joblib"

model = joblib.load(MODEL_PATH)
processor, classifier = model
explainer = shap.TreeExplainer(classifier)

GENDER_MAP = {'Male': 0, 'Female': 1}
GEO_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}

def preprocess_data(df: pd.DataFrame):

    df = df.copy()

    # Check df has necessary features
    required_cols = ["CreditScore", "Geography", "Gender", "Age", "Tenure", 
                     "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", 
                     "EstimatedSalary"] 

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    else:
        df = df[required_cols]

    # Encode Gender and Geography
    df['Gender'] = df['Gender'].map(GENDER_MAP).fillna(-1)
    df['Geography'] = df['Geography'].map(GEO_MAP).fillna(-1)

    return df

@api.post("/predict")
def predict_churn(data: ChurnPredictionInput):

    # Load input
    input_data = data.model_dump()
    df = pd.DataFrame([input_data])
    df = preprocess_data(df)

    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
   
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0]),
        "shap_values": get_shap_values(explainer, processor, df) if data.compute_shap else None
    }

@api.post("/predict-batch")
async def batch_predict_with_shap(file: UploadFile=File(...)):

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read file contents
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")
    
    df = preprocess_data(df)
    
    # Make predictions
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    
    result_df = df.copy()
    result_df['churn_prediction'] = predictions.astype(int)
    result_df['churn_probability'] = probabilities
    
    return {
        "predictions": result_df.to_dict(orient='records'),
        "shap_plot": make_beeswarm_viz(explainer, processor, df),
        "total_rows": len(result_df),
        "churn_count": int(predictions.sum())
    }

@api.get("/")
def read_root():
    return {"message": "Bank Churn API is running. Go to /docs to test it."}