from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the preprocessed model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all inputs in the correct order
        features = [
            float(request.form['SeniorCitizen']),
            float(request.form['tenure']),
            float(request.form['MonthlyCharges']),
            float(request.form['TotalCharges']),
            float(request.form['gender_Male']),
            float(request.form['Partner_Yes']),
            float(request.form['Dependents_Yes']),
            float(request.form['PhoneService_Yes']),
            float(request.form['MultipleLines_No phone service']),
            float(request.form['MultipleLines_Yes']),
            float(request.form['InternetService_Fiber optic']),
            float(request.form['InternetService_No']),
            float(request.form['OnlineSecurity_No internet service']),
            float(request.form['OnlineSecurity_Yes']),
            float(request.form['OnlineBackup_No internet service']),
            float(request.form['OnlineBackup_Yes']),
            float(request.form['DeviceProtection_No internet service']),
            float(request.form['DeviceProtection_Yes']),
            float(request.form['TechSupport_No internet service']),
            float(request.form['TechSupport_Yes']),
            float(request.form['StreamingTV_No internet service']),
            float(request.form['StreamingTV_Yes']),
            float(request.form['StreamingMovies_No internet service']),
            float(request.form['StreamingMovies_Yes']),
            float(request.form['Contract_One year']),
            float(request.form['Contract_Two year']),
            float(request.form['PaperlessBilling_Yes']),
            float(request.form['PaymentMethod_Credit card (automatic)']),
            float(request.form['PaymentMethod_Electronic check']),
            float(request.form['PaymentMethod_Mailed check'])
        ]

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]

        return render_template('index.html', prediction_text=f"Churn Prediction: {prediction}")

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)