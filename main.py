from flask import Flask, request
from ultralytics import YOLO
from PIL import Image
import io
import torch
import paho.mqtt.client as mqtt
import json
import traceback
from datetime import datetime
import os
from pathlib import Path

app = Flask(__name__)
model = YOLO("yolo11x-cls.pt")

MQTT_BROKER = '192.168.225.34'
MQTT_PORT = 1883
MQTT_TOPIC = 'esp32/sensors'

mqtt_client = None

RECEIVED_FOLDER = Path(os.path.dirname(os.path.abspath(__file__))) / '../received'

DANGEROUS_OBJECTS = {
    "assault rifle": 0.8,
    "rifle": 0.7,
    "jeep": 0.6,
    "car": 0.6,
    "truck":  0.6,
    "gun": 0.5,
    "revolver": 0.4,
    "person": 0.3
}

def ensure_received_folder_exists():
    if not RECEIVED_FOLDER.exists():
        RECEIVED_FOLDER.mkdir(parents=True)
        print(f"Created images directory: {RECEIVED_FOLDER}")
    else:
        print(f"Images directory already exists: {RECEIVED_FOLDER}")

def compute_danger(tags):
    danger = 0.0
    for tag in tags:
        if tag in DANGEROUS_OBJECTS:
            danger += DANGEROUS_OBJECTS[tag]
    return min(danger, 1.0)

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")

def on_disconnect(client, userdata, rc):
    print(f"Disconnected from MQTT broker with result code {rc}")

def setup_mqtt():
    global mqtt_client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"MQTT connection error: {e}")

def publish_sensor_data(danger):
    global mqtt_client
    if mqtt_client is None:
        setup_mqtt()

    try:
        payload = json.dumps({"camera": danger})
        mqtt_client.publish(MQTT_TOPIC, payload)
        print("MQTT publish result:", payload)
    except Exception as e:
        print(f"MQTT publish error: {e}")
        setup_mqtt()

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return {"error": "No image file provided"}, 400

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.jpg"
        file_path = RECEIVED_FOLDER / filename

        results = model(image)
        r = results[0]

        t5 = r.probs.top5
        tags = [model.names[i] for i in t5]

        danger_score = compute_danger(tags)
        if danger_score >= 0:
            image.save(file_path)
            print(f"Image saved to {file_path}")
            publish_sensor_data(danger_score)

        print(f"Tags: {tags}, Danger Score: {danger_score}")
        return {"tags": tags, "danger_score": danger_score, "saved_as": filename}

        results = model(image)
        r = results[0]

        t5 = r.probs.top5
        tags = [model.names[i] for i in t5]

        danger_score = compute_danger(tags)
        if danger_score > 0.1:
            publish_sensor_data(danger_score)

        print(f"Tags: {tags}, Danger Score: {danger_score}")
        return {"tags": tags, "danger_score": danger_score, "saved_as": filename}
    except Exception as e:
        print("Error during image processing:", e)
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == '__main__':
    ensure_received_folder_exists()
    setup_mqtt()
    app.run(host="0.0.0.0", port=5000)