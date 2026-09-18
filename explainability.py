import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence
from src.utils.logger import get_logger

logger = get_logger("model_explainability")

def get_feature_importances(model, feature_names: list) -> pd.DataFrame:
    """Extracts normalized feature importances from a trained tree/linear model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        importances = np.zeros(len(feature_names))
    
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df_imp

def compute_shap_summary(model, X_sample: pd.DataFrame) -> dict:
    """
    Computes SHAP values and returns average impact of each feature on PM2.5 predictions.
    """
    logger.info("Computing SHAP values for explainability...")
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        
        # Aggregate mean absolute SHAP values per feature
        if len(shap_values.values.shape) == 3: # multi-class
            vals = np.abs(shap_values.values).mean(axis=(0, 2))
        else:
            vals = np.abs(shap_values.values).mean(axis=0)
            
        shap_df = pd.DataFrame({
            "Feature": X_sample.columns,
            "SHAP_Impact": vals
        }).sort_values("SHAP_Impact", ascending=False).reset_index(drop=True)
        
        return {
            "shap_df": shap_df,
            "shap_values_raw": shap_values.values
        }
    except Exception as e:
        logger.warning(f"Error computing SHAP values: {e}. Fallback to feature importance.")
        return {"shap_df": None, "shap_values_raw": None}

def compute_partial_dependence(model, X: pd.DataFrame, feature_name: str, grid_resolution: int = 20) -> dict:
    """
    Computes Partial Dependence of PM2.5 prediction over a single meteorological feature.
    Shows how changing a specific weather feature (e.g. Temperature, Humidity) alters pollutant levels.
    """
    if feature_name not in X.columns:
        raise ValueError(f"Feature {feature_name} not present in input columns.")
    
    pd_results = partial_dependence(
        estimator=model,
        X=X,
        features=[feature_name],
        grid_resolution=grid_resolution
    )
    
    grid_values = pd_results["grid"][0]
    average_predictions = pd_results["average"][0]
    
    return {
        "feature": feature_name,
        "values": grid_values.tolist(),
        "predicted_pm25": average_predictions.tolist()
    }
