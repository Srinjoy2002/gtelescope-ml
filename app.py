from flask import Flask, request

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump, load

app = Flask(__name__)

# Load the dataset
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)

# Convert 'g' to 1 and 'h' to 0
df["class"] = (df["class"] == "g").astype(int)

# Scale the data
def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (len(y), 1))))

    return data, x, y, scaler  # Include scaler in the return

train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

train, x_train, y_train, scaler = scale_dataset(train, oversample=True)
valid, x_valid, y_valid, _ = scale_dataset(valid, oversample=False)
test, x_test, y_test, _ = scale_dataset(test, oversample=False)

# Train the SVM model
svm_model = SVC()
svm_model = svm_model.fit(x_train, y_train)

# Save the scaler and SVM model for later use
scaler_filename = "scaler.joblib"
model_filename = "svm_model.joblib"

dump(scaler, scaler_filename)
dump(svm_model, model_filename)

# Render the form using an inline template
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Extract user input from the form
        user_input = {
            "fLength": float(request.form["fLength"]),
            "fWidth": float(request.form["fWidth"]),
            "fSize": float(request.form["fSize"]),
            "fConc": float(request.form["fConc"]),
            "fConc1": float(request.form["fConc1"]),
            "fAsym": float(request.form["fAsym"]),
            "fM3Long": float(request.form["fM3Long"]),
            "fM3Trans": float(request.form["fM3Trans"]),
            "fAlpha": float(request.form["fAlpha"]),
            "fDist": float(request.form["fDist"]),
        }

        # Make a prediction
        predicted_class = predict_particle_class(user_input, scaler, svm_model)

        # Display the result
        return f"""
        <html>
            <head>
                <title>Particle Classification</title>
            </head>
            <body>
                <h1>Particle Classification</h1>
                <p>The predicted class is: {predicted_class}</p>
                <a href="/">Go back</a>
            </body>
        </html>
        """

    # Render the form
    return """
    <html>
        <head>
            <title>Particle Classification</title>
        </head>
        <body>
            <h1>Particle Classification</h1>
            <form method="post" action="/">
                <label for="fLength">fLength:</label>
                <input type="text" name="fLength" required><br>
                <label for="fWidth">fWidth:</label>
                <input type="text" name="fWidth" required><br>
                <label for="fSize">fSize:</label>
                <input type="text" name="fSize" required><br>
                <label for="fConc">fConc:</label>
                <input type="text" name="fConc" required><br>
                <label for="fConc1">fConc1:</label>
                <input type="text" name="fConc1" required><br>
                <label for="fAsym">fAsym:</label>
                <input type="text" name="fAsym" required><br>
                <label for="fM3Long">fM3Long:</label>
                <input type="text" name="fM3Long" required><br>
                <label for="fM3Trans">fM3Trans:</label>
                <input type="text" name="fM3Trans" required><br>
                <label for="fAlpha">fAlpha:</label>
                <input type="text" name="fAlpha" required><br>
                <label for="fDist">fDist:</label>
                <input type="text" name="fDist" required><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

def predict_particle_class(user_input, scaler, model):
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)
    predicted_class = "gamma" if prediction[0] == 1 else "hadron"
    return predicted_class

if __name__ == "__main__":
    app.run(debug=True)
