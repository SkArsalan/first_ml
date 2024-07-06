from flask import Flask, request, jsonify
import joblib
from flasgger import Swagger
from flasgger.utils import swag_from
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

swagger = Swagger(app, template_file='swagger_config.yaml')

@app.route("/")
@app.route("/home")
@swag_from('swagger_config.yaml', methods=['GET'])
def home():
    return "<h1>This is Home Page</h1>"

@app.route("/predict", methods=['POST'])
@swag_from('swagger_config.yaml', methods=['POST'])
def predict():
    msg = request.json
    value = float(msg['Num1'])
    load_poly_reg = joblib.load('poly_features.pkl')
    load_lin_reg = joblib.load('poly_model.pkl')
    poly_prediction = load_lin_reg.predict(load_poly_reg.fit_transform([[value]]))
    prediction = int(poly_prediction[0])  # Extract the prediction value
    print("Your message:", prediction)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
