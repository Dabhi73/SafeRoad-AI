import streamlit as st
import os
import json
import time
import joblib
import ollama
from google import genai
from pydantic import BaseModel, Field

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="SafeRoad AI - Dual Engine Comparison", page_icon="🚦", layout="wide")

# --- 2. LOAD TRAINED PRIORITY ML MODEL ---
@st.cache_resource
def load_priority_ml_model():
    if os.path.exists("priority_model.pkl"):
        return joblib.load("priority_model.pkl")
    return None

ml_priority_model = load_priority_ml_model()

# Historical risk lookup dictionary
historical_ml_database = {
    "outside the university gate": {"historical_risk": "HIGH", "past_accidents": 14},
    "sector 4 park": {"historical_risk": "LOW", "past_accidents": 1},
    "sudama chokdi": {"historical_risk": "HIGH", "past_accidents": 8}
}

# --- 3. STRUCTURED OUTPUT SCHEMA ---
class ComplaintAssessment(BaseModel):
    Category: str = Field(description="e.g., Road Infrastructure, Lighting, Traffic, Waterlogging")
    Location: str = Field(description="Extract the specific location mentioned")
    Severity: str = Field(description="LOW, MEDIUM, or HIGH based on risk to life/property")
    Incident: str = Field(description="Briefly describe any accidents or events mentioned")
    Department: str = Field(description="Which authority should handle this")
    Priority: str = Field(description="LOW, MEDIUM, or HIGH")
    Recommended_Action: str = Field(description="A short, actionable step for the authority")

# --- 4. BACKEND INFERENCE FUNCTIONS ---

def analyze_with_gemini(complaint_text: str):
    start_time = time.time()
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
    elapsed = round(time.time() - start_time, 2)
    return response.parsed, elapsed

def analyze_with_ollama(complaint_text: str):
    start_time = time.time()
    system_instruction = "You are an AI assistant for a city's road safety department. Extract the data accurately."
    prompt = f"{system_instruction}\n\nAnalyze this citizen complaint:\n'{complaint_text}'"
    
    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        format=ComplaintAssessment.model_json_schema(),
        options={'temperature': 0}
    )
    
    parsed_data = ComplaintAssessment.model_validate_json(response.message.content)
    elapsed = round(time.time() - start_time, 2)
    return parsed_data, elapsed

# --- 5. FUSION LOGIC ---
def calculate_final_action(complaint_json, full_text_input):
    loc_key = complaint_json.Location.lower().strip()
    hist_risk = "MEDIUM"
    for k in historical_ml_database:
        if k in loc_key or loc_key in k:
            hist_risk = historical_ml_database[k]["historical_risk"]
            break

    # Predict Priority using the Trained ML Model (.pkl)
    if ml_priority_model:
        predicted_ml_priority = ml_priority_model.predict([full_text_input])[0]
    else:
        predicted_ml_priority = complaint_json.Priority.upper()
        
    ai_priority = complaint_json.Priority.upper()
    
    if hist_risk == "HIGH" and (ai_priority == "HIGH" or predicted_ml_priority == "HIGH"):
        final_urgency = "🔴 CRITICAL"
    elif hist_risk == "HIGH" or ai_priority == "HIGH" or predicted_ml_priority == "HIGH":
        final_urgency = "🟠 HIGH"
    else:
        final_urgency = "🟢 ROUTINE"
        
    return hist_risk, predicted_ml_priority, final_urgency

# --- 6. FRONTEND DASHBOARD UI ---
st.title("🚦 SafeRoad AI: Cloud vs. Local Model Comparison")
st.markdown("Fusing a **Trained ML Priority Classifier** with **Generative AI NLP Extraction**.")
st.divider()

st.subheader("📥 Incoming Citizen Complaint")
default_text = "The road outside the university gate has a very deep pothole. There is no streetlight and two bikes slipped there yesterday."
user_input = st.text_area("Live Report Feed:", value=default_text, height=100)

if st.button("Run Model Comparison", type="primary"):
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("Gemini API Key missing! Set your GEMINI_API_KEY variable in the terminal first.")
    else:
        with st.spinner("Processing complaint across Cloud and Local LLM engines simultaneously..."):
            try:
                gemini_data, gemini_time = analyze_with_gemini(user_input)
                gemini_hist_risk, gemini_ml_prio, gemini_urgency = calculate_final_action(gemini_data, user_input)
                
                ollama_data, ollama_time = analyze_with_ollama(user_input)
                ollama_hist_risk, ollama_ml_prio, ollama_urgency = calculate_final_action(ollama_data, user_input)

                st.success("Analysis Complete!")
                st.divider()

                col_cloud, col_local = st.columns(2)

                # Column 1: Cloud Gemini
                with col_cloud:
                    st.header("☁️ Cloud Model (Gemini 3.6 Flash)")
                    st.caption(f"⚡ Latency: **{gemini_time} seconds**")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Hist. Risk", gemini_hist_risk)
                    m2.metric("Trained ML Priority", gemini_ml_prio)
                    m3.metric("GenAI Priority", gemini_data.Priority)
                    m4.metric("Status", gemini_urgency.split()[-1])
                    
                    st.write(f"**📍 Location:** {gemini_data.Location}")
                    st.write(f"**⚠️ Category:** {gemini_data.Category}")
                    st.write(f"**🚨 Incidents:** {gemini_data.Incident}")
                    st.write(f"**🏢 Department:** {gemini_data.Department}")
                    st.info(f"**🛠️ Recommended Action:** {gemini_data.Recommended_Action}")
                    
                    with st.expander("Gemini Raw JSON"):
                        st.json(gemini_data.model_dump())

                # Column 2: Local Ollama
                with col_local:
                    st.header("💻 Local Model (Llama 3.1)")
                    st.caption(f"⚡ Latency: **{ollama_time} seconds**")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Hist. Risk", ollama_hist_risk)
                    m2.metric("Trained ML Priority", ollama_ml_prio)
                    m3.metric("GenAI Priority", ollama_data.Priority)
                    m4.metric("Status", ollama_urgency.split()[-1])
                    
                    st.write(f"**📍 Location:** {ollama_data.Location}")
                    st.write(f"**⚠️ Category:** {ollama_data.Category}")
                    st.write(f"**🚨 Incidents:** {ollama_data.Incident}")
                    st.write(f"**🏢 Department:** {ollama_data.Department}")
                    st.info(f"**🛠️ Recommended Action:** {ollama_data.Recommended_Action}")
                    
                    with st.expander("Ollama Raw JSON"):
                        st.json(ollama_data.model_dump())

            except Exception as e:
                st.error(f"Execution Error: {e}")