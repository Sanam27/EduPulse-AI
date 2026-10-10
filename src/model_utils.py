"""
Model Utilities for EduPulse AI
Handles model loading, inference, risk probability calculation, and feature indicator explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/risk_model.pkl"

FEATURE_NAMES_REMAPPED = {
    "attendance_rate": "Attendance Rate (%)",
    "gpa": "GPA (4.0 Scale)",
    "assignment_completion_rate": "Assignment Completion (%)",
    "midterm_score": "Midterm Score (%)",
    "study_hours_per_week": "Study Hours / Week",
    "lms_engagement_score": "LMS Portal Engagement",
    "participation_score": "Class Participation",
    "stress_level": "Stress Level (1-10)"
}

def load_model(filepath=MODEL_PATH):
    """Load model artifact dictionary from pickle file."""
    if not os.path.exists(filepath):
        from train_model import train_risk_model
        return train_risk_model()
        
    artifact = joblib.load(filepath)
    return artifact

def predict_student_risk(student_data, artifact=None):
    """
    Predict risk level and probability distribution for a student record.
    student_data can be a pandas Series or dictionary containing feature values.
    """
    if artifact is None:
        artifact = load_model()
        
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    # Prepare feature vector
    if isinstance(student_data, dict):
        input_df = pd.DataFrame([student_data])[feature_cols]
    else:
        input_df = pd.DataFrame([student_data[feature_cols].to_dict()])[feature_cols]
        
    predicted_class = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_dict = dict(zip(model.classes_, probabilities))
    
    return {
        "predicted_risk": predicted_class,
        "probabilities": prob_dict,
        "high_risk_prob": round(prob_dict.get("High", 0.0) * 100, 1),
        "medium_risk_prob": round(prob_dict.get("Medium", 0.0) * 100, 1),
        "low_risk_prob": round(prob_dict.get("Low", 0.0) * 100, 1)
    }

def explain_student_indicators(student_data, artifact=None):
    """
    Analyze key indicators contributing to a student's risk prediction.
    Compares student metrics against cohort averages, weighted by Random Forest feature importance.
    """
    if artifact is None:
        artifact = load_model()
        
    feature_cols = artifact["features"]
    importances = artifact["feature_importances"]
    cohort_stats = artifact["cohort_stats"]
    
    indicators = []
    
    for feat in feature_cols:
        val = float(student_data[feat])
        mean_val = cohort_stats[feat]["mean"]
        std_val = cohort_stats[feat]["std"] if cohort_stats[feat]["std"] > 0 else 1.0
        imp = importances.get(feat, 0.1)
        
        # Stress level: higher is worse; for others: lower is worse
        if feat == "stress_level":
            diff = val - mean_val
            z_score = diff / std_val
            is_risk = z_score > 0.5
            is_strength = z_score < -0.5
        else:
            diff = mean_val - val
            z_score = (val - mean_val) / std_val
            is_risk = z_score < -0.5
            is_strength = z_score > 0.5
            
        impact_score = abs(z_score) * imp * 100
        
        if is_risk:
            status = "Risk Driver"
            badge = "🔴 Critical" if abs(z_score) > 1.2 else "🟡 Concern"
        elif is_strength:
            status = "Positive Strength"
            badge = "🟢 Strong"
        else:
            status = "Average"
            badge = "⚪ Normal"
            
        pretty_name = FEATURE_NAMES_REMAPPED.get(feat, feat)
        
        indicators.append({
            "feature_key": feat,
            "feature_name": pretty_name,
            "student_value": val,
            "cohort_avg": round(mean_val, 1),
            "z_score": round(z_score, 2),
            "importance": round(imp * 100, 1),
            "impact_score": round(impact_score, 1),
            "status": status,
            "badge": badge
        })
        
    # Sort indicators by impact score descending
    indicators.sort(key=lambda x: x["impact_score"], reverse=True)
    return indicators
