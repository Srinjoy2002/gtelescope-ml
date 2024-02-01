import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump, load

# Load the dataset
pd.read_csv("magic04.data")

# Define columns and read the dataset
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)

# Convert 'g' to 1 and 'h' to 0
df["class"] = (df["class"] == "g").astype(int)

# Visualize the data using bar graphs
for label in cols[:-1]:
    plt.hist(df[df["class"] == 1][label], color='blue', label='gamma', alpha=0.6, density=True)
    plt.hist(df[df["class"] == 0][label], color='red', label='hadron', alpha=0.6, density=True)
    plt.title(label)
    plt.ylabel("probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

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

# Example user input
user_input = {
    "fLength": 10.0,
    "fWidth": 20.0,
    "fSize": 5.0,
    "fConc": 0.1,
    "fConc1": 0.2,
    "fAsym": 0.3,
    "fM3Long": 0.4,
    "fM3Trans": 0.5,
    "fAlpha": 0.6,
    "fDist": 0.7
}

# Load the scaler and SVM model
scaler = load(scaler_filename)
svm_model = load(model_filename)

# Make a prediction using the function
predicted_class = predict_particle_class(user_input, scaler, svm_model)

# Display the result
print(f"The predicted class is: {predicted_class}")
