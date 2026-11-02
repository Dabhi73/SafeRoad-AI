import os
import json
import joblib
from google import genai
from pydantic import BaseModel, Field

# Load trained Priority ML model
try:
    priority_ml_model = joblib.load("priority_model.pkl")
    print("✅ Loaded trained Priority ML model successfully.")
except Exception as e:
    priority_ml_model = None
    print("⚠️ Warning: Could not load priority ML model.")

class ComplaintAssessment(BaseModel):
    Category: str = Field(description="e.g., Road Infrastructure, Lighting, Traffic, Waterlogging")
    Location: str = Field(description="Extract the specific location mentioned")
    Severity: str = Field(description="LOW, MEDIUM, or HIGH based on risk to life/property")
    Incident: str = Field(description="Briefly describe any accidents or events mentioned")
    Department: str = Field(description="Which authority should handle this")
    Priority: str = Field(description="LOW, MEDIUM, or HIGH")
    Recommended_Action: str = Field(description="A short, actionable step for the authority")

historical_ml_database = {
    "outside the university gate": {"historical_risk": "HIGH", "past_accidents": 14},
    "sector 4 park": {"historical_risk": "LOW", "past_accidents": 1},
}

def analyze_complaint(complaint_text: str):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    system_instruction = "You are an AI assistant for a city's road safety department. Extract the data accurately."
    prompt = f"{system_instruction}\n\nAnalyze this citizen complaint:\n'{complaint_text}'"
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ComplaintAssessment,
        }
    )
    return response.parsed

def calculate_final_action(complaint_json, full_text_input):
    loc_key = complaint_json.Location.lower().strip()
    ml_data = historical_ml_database.get(loc_key, {"historical_risk": "MEDIUM", "past_accidents": "Unknown"})
    hist_risk = ml_data["historical_risk"]
    
    if priority_ml_model:
        ml_priority = priority_ml_model.predict([full_text_input])[0]
    else:
        ml_priority = complaint_json.Priority
        
    ai_priority = complaint_json.Priority
    
    if hist_risk == "HIGH" and (ai_priority == "HIGH" or ml_priority == "HIGH"):
        final_urgency = "CRITICAL / IMMEDIATE DISPATCH"
    elif hist_risk == "HIGH" or ai_priority == "HIGH" or ml_priority == "HIGH":
        final_urgency = "HIGH PRIORITY"
    else:
        final_urgency = "ROUTINE MAINTENANCE"
        
    return {
        "Location": complaint_json.Location,
        "Issue": complaint_json.Category,
        "Historical_Accident_Risk": hist_risk,
        "ML_Trained_Priority": ml_priority,
        "GenAI_Extracted_Priority": ai_priority,
        "Final_Urgency_Level": final_urgency,
        "Assigned_Department": complaint_json.Department,
        "Action_Required": complaint_json.Recommended_Action
    }

if __name__ == "__main__":
    sample_complaint = (
        "The road outside the university gate has a very deep pothole. "
        "There is no streetlight and two bikes slipped there yesterday."
    )
    
    try:
        assessment = analyze_complaint(sample_complaint)
        final_decision = calculate_final_action(assessment, sample_complaint)
        print("--- FINAL AUTHORITY DASHBOARD TICKET ---")
        print(json.dumps(final_decision, indent=2))
    except Exception as e:
        print(f"Error: {e}")