import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
 
# 1. Dataset mapping complaint text to Priority levels
dataset = {
    "complaint_text": [
        "Deep pothole on main road causing severe bike accidents",
        "Streetlight completely off creating dark hazardous driving conditions",
        "Open manhole cover on busy pedestrian sidewalk near school",
        "Waterlogging on highway blocking entire traffic flow",
        "Multiple vehicles slipped due to oil spill near university gate",
        "Minor garbage pile near residential park gate",
        "Faded zebra crossing marks near neighborhood market",
        "Tree branch drooping slightly over side pavement",
        "Small pothole near quiet residential avenue",
        "Traffic signal timer slightly out of sync",
        "Broken street lamp in residential lane",
        "Water accumulation near street pavement after minor rain"
    ],
    "priority": [
        "HIGH", "HIGH", "HIGH", "HIGH", "HIGH",
        "LOW", "LOW", "LOW", "LOW",
        "MEDIUM", "MEDIUM", "MEDIUM"
    ]
}

# 2. Save dataset to CSV for submission to your instructor
df = pd.DataFrame(dataset)
df.to_csv("complaint_priority_data.csv", index=False)
print("✅ Created 'complaint_priority_data.csv'")

# 3. Create an NLP Vectorizer + Random Forest Model Pipeline
pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# 4. Train the model (Input = Text, Output = Priority)
pipeline.fit(df["complaint_text"], df["priority"])

# 5. Save the trained ML model file
joblib.dump(pipeline, "priority_model.pkl")
print("✅ Trained ML model saved as 'priority_model.pkl'")