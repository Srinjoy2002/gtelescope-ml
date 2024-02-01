import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Function to preprocess user input and make a prediction
def predict_particle_class(user_input, scaler, model):
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input])

    # Scale user input using the same scaler
    user_scaled = scaler.transform(user_df)

    # Make a prediction using the trained model
    prediction = model.predict(user_scaled)

    # Convert the prediction to a human-readable class
    predicted_class = "gamma" if prediction > 0.5 else "hadron"

    return predicted_class

# Load the trained model and scaler
least_loss_model = tf.keras.models.load_model("magic_ds.py")  # Replace with the actual path
scaler = StandardScaler()  # Make sure to use the same scaler used during training

# Example user input (replace with actual user input)
user_input = {
    "fLength": 10.0,
    "fWidth": 20.0,
    "fSize": 5.0
}

# Make a prediction using the function
predicted_class = predict_particle_class(user_input, scaler, least_loss_model)

# Display the result
print(f"The predicted class is: {predicted_class}")
