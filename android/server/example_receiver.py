"""
Example receiver for the DefectCam Android app.

Multiple phones POST JPEG frames here concurrently, each tagged with a
device_id. This script decodes each frame to an image and calls a stub
`run_inference` where the real defect-detection model plugs in.

Run:
    pip install flask opencv-python numpy
    python example_receiver.py

Then point the app's in-app "server URL" field at:
    http://<this-machine's-LAN-IP>:5000/upload
"""

import cv2
import numpy as np
from flask import Flask, request

app = Flask(__name__)


def run_inference(image: np.ndarray, device_id: str, timestamp_ms: str) -> dict:
    """Replace this with your real model call."""
    # result = your_model.predict(image)
    return {"device_id": device_id, "timestamp": timestamp_ms, "defect": None}


def push_to_dashboard(result: dict) -> None:
    """Replace with however you fan results out to the dashboard
    (WebSocket, Flask-SocketIO broadcast, DB write, message queue, ...)."""
    print(f"[dashboard] {result}")


@app.route("/upload", methods=["POST"])
def upload():
    device_id = request.form.get("device_id", "unknown")
    timestamp_ms = request.form.get("timestamp", "")
    frame_file = request.files.get("frame")

    if frame_file is None:
        return "missing 'frame' file field", 400

    jpeg_bytes = frame_file.read()
    image_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return "could not decode frame", 400

    result = run_inference(image, device_id, timestamp_ms)
    push_to_dashboard(result)

    return "", 200


if __name__ == "__main__":
    # host="0.0.0.0" is required so phones on the LAN can reach this,
    # not just the laptop itself.
    app.run(host="0.0.0.0", port=5000)
