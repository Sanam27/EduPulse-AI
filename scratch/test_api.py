"""
FastAPI Test Script for EduPulse AI
Tests GET /health, GET /, and POST /predict using FastAPI TestClient.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from api import app

def run_fastapi_tests():
    client = TestClient(app)

    print("==================================================")
    print("EduPulse AI - FastAPI Backend Test Suite")
    print("==================================================\n")

    # Test 1: Root GET /
    res_root = client.get("/")
    print("[TEST 1] GET / Endpoint:")
    print(f"  - Status Code: {res_root.status_code}")
    print(f"  - Response JSON: {res_root.json()}")
    assert res_root.status_code == 200, "Root endpoint failed"
    print("  => STATUS: PASS\n")

    # Test 2: GET /health
    res_health = client.get("/health")
    print("[TEST 2] GET /health Endpoint:")
    print(f"  - Status Code: {res_health.status_code}")
    print(f"  - Response JSON: {res_health.json()}")
    assert res_health.status_code == 200, "Health check failed"
    assert res_health.json()["status"] == "healthy", "Model not healthy"
    print("  => STATUS: PASS\n")

    # Test 3: POST /predict (Valid Request)
    payload = {
        "student_id": "EDU-TEST-999",
        "name": "Sarah Connor",
        "major": "Computer Science",
        "grade_level": "Junior",
        "attendance_rate": 62.0,
        "gpa": 2.1,
        "assignment_completion_rate": 58.0,
        "midterm_score": 55.0,
        "study_hours_per_week": 8.0,
        "lms_engagement_score": 45.0,
        "participation_score": 50.0,
        "stress_level": 8,
        "tuition_status": "Pending"
    }

    res_pred = client.post("/predict", json=payload)
    print("[TEST 3] POST /predict Endpoint (High Risk Sample):")
    print(f"  - Status Code: {res_pred.status_code}")
    data = res_pred.json()
    print(f"  - Predicted Risk: {data['predicted_risk']}")
    print(f"  - High Risk Prob: {data['high_risk_prob']}%")
    print(f"  - Success Score: {data['success_score']}/100")
    print(f"  - Placement Risk: {data['placement_risk']}")
    print(f"  - Top Recommendation: {data['recommendations'][0]['action']}")
    assert res_pred.status_code == 200, "Predict endpoint failed"
    assert data["predicted_risk"] in ["High", "Medium", "Low"], "Invalid risk class"
    print("  => STATUS: PASS\n")

    # Test 4: POST /predict (Invalid Validation Request)
    invalid_payload = {
        "attendance_rate": 150.0,  # Invalid: > 100
        "gpa": 5.0                 # Invalid: > 4.0
    }

    res_invalid = client.post("/predict", json=invalid_payload)
    print("[TEST 4] POST /predict Endpoint (Invalid Inputs Pydantic Validation):")
    print(f"  - Status Code: {res_invalid.status_code} (Expected 422 Unprocessable Entity)")
    assert res_invalid.status_code == 422, "Pydantic validation failed"
    print("  => STATUS: PASS\n")

    print("==================================================")
    print("ALL 4 FASTAPI BACKEND TESTS PASSED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    run_fastapi_tests()
