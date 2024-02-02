import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump, load

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

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

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

# Function to preprocess user input and make a prediction
def predict_particle_class(user_input, scaler, model):
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)
    predicted_class = "gamma" if prediction[0] == 1 else "hadron"
    return predicted_class
