from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
try:
    with open("heart_disease_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise FileNotFoundError("Model file 'heart_disease_model.pkl' not found. Ensure the file is in the same directory.")
except Exception as e:
    raise Exception(f"Error loading model file: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract features from the input JSON
        features = np.array([
            data["age"], data["sex"], data["cp"], data["trestbps"],
            data["chol"], data["fbs"], data["restecg"], data["thalach"],
            data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"]
        ]).reshape(1, -1)

        # Make predictions using the loaded model
        prediction = model.predict(features)[0]

        # Return a readable prediction result
        result = "Positive for Heart Disease" if prediction == 1 else "Negative for Heart Disease"
        return jsonify({"prediction": result})

    except KeyError as e:
        return jsonify({"error": f"Missing key in input data: {str(e)}. Ensure all fields are provided."})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
