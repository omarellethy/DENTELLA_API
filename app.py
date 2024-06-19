import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    packages = [
        "flask",
        "flask_restful",
        "keras",
        "pillow",  # PIL is now Pillow
        "numpy",
        "python-dotenv",
        "pyngrok"
    ]

    for package in packages:
        try:
            install(package)
            print(f"{package} has been installed successfully.")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import keras
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from pyngrok import ngrok
import os

load_dotenv()

classes = [
    "Your Teeth are Bad & You Need To Visit Doctor",
    "Your Teeth are Good & You Don't Need To Visit Doctor"
]

# Ensure the model path is correct
model_path = "APP-host/code/models/imageclassifier.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = keras.models.load_model(model_path)

def predict(img):
    img = img.resize((224, 224))
    img_np = np.array(img) / 255.0
    img_np = img_np.reshape(1, 224, 224, 3)
    pred = model.predict(img_np, verbose=0)
    return classes[np.argmax(pred)]  # Use np.argmax to get the index of the highest score

class Predict(Resource):
    def post(self):
        file = request.files['image']
        img = Image.open(file)
        return {"result": predict(img)}

app = Flask(__name__)
api = Api(app)
api.add_resource(Predict, '/classify')

NGROK_AUTH = os.getenv("NGROK_AUTH")
if not NGROK_AUTH:
    raise ValueError("NGROK_AUTH environment variable not set")

print(f"NGROK_AUTH: {NGROK_AUTH}")  # Debug print to verify the auth token
PORT = 5000

ngrok.set_auth_token(NGROK_AUTH)
tunnel = ngrok.connect(PORT, domain="exciting-mammoth-carefully.ngrok-free.app")
print("Public URL:", tunnel.public_url)
app.run(host='0.0.0.0', port=5000)
