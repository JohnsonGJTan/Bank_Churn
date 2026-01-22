from pydantic import BaseModel, Field

class ChurnPredictionInput(BaseModel):
    CreditScore: int
    Geography: str  
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    compute_shap: bool = Field(default=False, description="Whether to compute SHAP values for explainability")