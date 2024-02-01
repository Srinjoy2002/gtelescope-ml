from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import load

app = Flask(__name__)

# Load the pre-trained model and scaler
scaler = load("scaler.joblib")
svm_model = load("svm_model.joblib")

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            "fLength": float(request.form['fLength']),
            "fWidth": float(request.form['fWidth']),
            "fSize": float(request.form['fSize']),
            "fConc": float(request.form['fConc']),
            "fConc1": float(request.form['fConc1']),
            "fAsym": float(request.form['fAsym']),
            "fM3Long": float(request.form['fM3Long']),
            "fM3Trans": float(request.form['fM3Trans']),
            "fAlpha": float(request.form['fAlpha']),
            "fDist": float(request.form['fDist'])
        }

        # Preprocess user input
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)

        # Make a prediction
        prediction = svm_model.predict(user_scaled)
        predicted_class = "gamma" if prediction[0] == 1 else "hadron"

        # Pass the prediction to the result page
        return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
