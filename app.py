from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs
        feature_order = [
            'SeniorCitizen', 'MonthlyCharges',
            'MultipleLines_No phone service',
            'InternetService_Fiber optic','StreamingTV_Yes',
            'StreamingMovies_Yes',
             'PaperlessBilling_Yes',
            'PaymentMethod_Electronic check','MultipleLines_Yes',
            'PaymentMethod_Mailed check'
        ]

        features = [float(request.form.get(feat, 0)) for feat in feature_order]
        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)[0]
        result = "Yes" if prediction == 1 else "No"

        return render_template('index.html', prediction_text=f"Churn Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
