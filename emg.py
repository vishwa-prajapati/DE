import streamlit as st
import pandas as pd
import joblib

# Load the trained model
rf_model = joblib.load("ahmedabad_traffic_model_rf.pkl")

# Title
st.title("🚦Emergency Traffic Control System")
st.markdown("Predict which lane should get the green light based on traffic and emergency input.")

# User Input Form
st.header("📥 Input Lane Data")

with st.form("lane_input_form"):
    traffic_inputs = {}
    green_time_inputs = {}
    emergency_inputs = {}

    for i in range(1, 5):
        traffic_inputs[f"Traffic_Lane_{i}"] = st.number_input(f"Traffic in Lane {i}", min_value=0, step=10)
        green_time_inputs[f"Green_Time_Lane_{i}"] = st.number_input(f"Previous Green Time for Lane {i}", min_value=0, step=5)
        emergency_inputs[f"Emergency_Lane_{i}"] = st.selectbox(f"Emergency in Lane {i}?", options=[0, 1], index=0, key=f"emg_{i}")

    submit = st.form_submit_button("🚥 Predict Green Lane")

# Prediction Function
def test_model_rf(test_input):
    df_test = pd.DataFrame([test_input])

    emergency_lanes = [i for i in range(1, 5) if test_input[f"Emergency_Lane_{i}"] == 1]
    if len(emergency_lanes) > 1:
        emergency_lanes.sort(key=lambda lane: test_input[f"Traffic_Lane_{lane}"], reverse=True)
        lane_green = emergency_lanes[0]
        message = f"🚨 Multiple emergencies detected! Prioritizing Lane {lane_green} with highest traffic."
    elif len(emergency_lanes) == 1:
        lane_green = emergency_lanes[0]
        message = f"🚨 Emergency vehicle detected! Giving green light to Lane {lane_green}."
    else:
        lane_green = round(rf_model.predict(df_test)[0])
        message = f"🚦 Lane {lane_green} will have the green light."
    return lane_green, message

# Prediction Result
if submit:
    # Combine all inputs
    test_input = {**traffic_inputs, **green_time_inputs, **emergency_inputs}
    lane_green, message = test_model_rf(test_input)

    st.subheader("✅ Prediction Result")
    st.success(message)
    st.markdown(f"**Predicted Lane:** {lane_green}")

    st.subheader("🔍 Input Summary")
    st.dataframe(pd.DataFrame([test_input]))

