import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model
with open("rnf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler model
with open("scaler.pkl", "rb") as model_file:
    scaler = pickle.load(model_file)

# Streamlit app
st.title("FIFA Sports Prediction")

# User input
potential = st.slider("Potential", min_value=0, max_value=100, value=50)
value_eur = st.number_input("Value Eur", min_value=0)
wage_eur = st.number_input("Wage Eur", min_value=0)
release_clause_eur = st.number_input(
    "Release Clause Eur", min_value=0, value=50
)
passing = st.slider("Passing", min_value=0, max_value=100, value=50)
dribbling = st.slider("Dribbling", min_value=0, max_value=100, value=50)
attacking_short_passing = st.slider(
    "Attackinh Short Passing", min_value=0, max_value=100, value=50
)
movement_reactions = st.slider(
    "Movement Reactions", min_value=0, max_value=100, value=50
)
power_shot_power = st.slider("Power Shot Power", min_value=0, max_value=100, value=50)
mentality_vision = st.slider("Mentality Vision", min_value=0, max_value=100, value=50)
mentality_composure = st.slider(
    "Mentality Composure", min_value=0, max_value=100, value=50
)

# Numpy Array
data = np.array(
    [[  potential,
        value_eur,
        wage_eur,
        release_clause_eur,
        passing,
        dribbling,
        attacking_short_passing,
        movement_reactions,
        power_shot_power,
        mentality_vision,
        mentality_composure,
    ]]
)
# Data Frame
df = pd.DataFrame(
    data,
    columns=[
        "potential",
        "value_eur",
        "wage_eur",
        "release_clause_eur",
        "passing",
        "dribbling",
        "attacking_short_passing",
        "movement_reactions",
        "power_shot_power",
        "mentality_vision",
        "mentality_composure",
    ]
)

# Scaling the data
scaled_df = scaler.transform(df)
# Make predictions
prediction = model.predict(scaled_df)[0]

# Display prediction
st.subheader("Prediction:")
st.write(f"The predicted output is: {prediction}")

# Additional content (if needed)
st.write(
    "This is a simple Streamlit web app to demonstrate a machine learning model prediction."
)

# You can add more UI elements and features to your app as needed.
