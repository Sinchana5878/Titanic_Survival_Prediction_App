import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model
model = pickle.load(open('titanic_model.pkl', 'rb'))

# Streamlit page setup
st.set_page_config(page_title="Titanic Survival Predictor 🚢", page_icon="🛟", layout="centered")

# Title and description
st.title("🚢 Titanic Survival Prediction App")
st.markdown("""
### 🌊 Predict your chance of survival on the Titanic!
Fill in the passenger details below, and this ML model will estimate your **survival probability**.
""")

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🎫 Passenger Class", [1, 2, 3])
    sex = st.selectbox("👤 Sex", ["Male", "Female"])
    age = st.number_input("🎂 Age (years)", min_value=0, max_value=100, value=25)
    embarked = st.selectbox("🛳️ Port of Embarkation", ["S", "C", "Q"])

with col2:
    sibsp = st.number_input("👫 Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("👨‍👩‍👧 Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("💰 Ticket Fare", min_value=0.0, value=32.0)

# Convert categorical data to numeric
sex = 0 if sex == "Male" else 1
embarked_map = {'S': 0, 'C': 1, 'Q': 2}
embarked = embarked_map[embarked]

# Create input DataFrame
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

st.markdown("---")

# Predict button
if st.button("🔍 Predict Survival"):
    prediction = model.predict(input_data)
    survival_prob = 0.0
    death_prob = 0.0

    # If model supports probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        death_prob, survival_prob = probs[0] * 100, probs[1] * 100
    else:
        # fallback if no proba support
        survival_prob = 100.0 if prediction[0] == 1 else 0.0
        death_prob = 100.0 - survival_prob

    # Display result
    st.markdown("## 🧾 Prediction Result")
    if prediction[0] == 1:
        st.success(f"🎉 **Survived!** (Survival Probability: {survival_prob:.2f}%)")
    else:
        st.error(f"💀 **Did Not Survive.** (Survival Probability: {survival_prob:.2f}%)")

    # Progress bar for survival probability
    st.progress(int(survival_prob))

    # Bar chart visualization
    st.markdown("### 📊 Survival vs Non-Survival Probability")
    fig, ax = plt.subplots(figsize=(5, 3))
    probs_dict = {"Not Survived": death_prob, "Survived": survival_prob}
    bars = ax.bar(probs_dict.keys(), probs_dict.values(), color=["#E74C3C", "#2ECC71"])
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.bar_label(bars, fmt='%.1f%%', label_type='edge', fontsize=10)
    ax.set_facecolor("#F9F9F9")
    st.pyplot(fig)



