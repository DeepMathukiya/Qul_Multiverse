"""
Receiver for the DefectCam Android app.

Architecture:

Android phones
      |
      | POST JPEG frames
      v
Flask /upload
      |
      | decode frame
      v
Store latest frame per device
      |
      +----> run_inference()
      |
      v
Main thread OpenCV display

Run:
    pip install flask opencv-python numpy

    python example_receiver.py
"""

import time
import threading

import cv2
import numpy as np

from flask import Flask, request


app = Flask(__name__)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

WINDOW_NAME = "Incoming stream"

FPS_WINDOW_SEC = 1.0


# --------------------------------------------------
# Shared Frame Storage
# --------------------------------------------------

latest_frames = {}

frame_lock = threading.Lock()


# --------------------------------------------------
# FPS Tracking (per device)
# --------------------------------------------------

# Each device gets its own window + FPS counters so
# that two (or more) phones streaming at once are
# displayed side by side instead of overwriting each
# other in a single shared window.

fps_start_time = {}

fps_frame_count = {}

last_fps = {}


def window_name_for(device_id: str) -> str:

    """OpenCV window title for a given device."""

    return f"{WINDOW_NAME} - {device_id}"


# --------------------------------------------------
# Store Frame
# --------------------------------------------------

def update_latest_frame(
    image: np.ndarray,
    device_id: str,
    timestamp_ms: str,
) -> None:

    """
    Store the latest frame received from each device.

    Thread-safe because multiple phones may upload
    frames simultaneously.
    """

    with frame_lock:

        latest_frames[device_id] = {
            "image": image.copy(),
            "timestamp": timestamp_ms,
        }


# --------------------------------------------------
# Run Inference
# --------------------------------------------------

def run_inference(
    image: np.ndarray,
    device_id: str,
    timestamp_ms: str,
) -> dict:

    """
    Replace this function with:

    Canny Edge Detection
            ↓
    Contour Extraction
            ↓
    ROI Detection
            ↓
    YOLO Defect Analysis
    """

    # Example:

    # result = model.predict(image)

    return {

        "device_id": device_id,

        "timestamp": timestamp_ms,

        "defect": None,

    }


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

def push_to_dashboard(result: dict) -> None:

    """
    Replace this with:

    WebSocket
    Flask-SocketIO
    Database
    Message Queue
    etc.
    """

    print(f"[dashboard] {result}")


# --------------------------------------------------
# Upload Endpoint
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():

    device_id = request.form.get(
        "device_id",
        "unknown",
    )

    timestamp_ms = request.form.get(
        "timestamp",
        "",
    )

    frame_file = request.files.get("frame")


    if frame_file is None:

        return "missing 'frame' file field", 400


    # Read JPEG

    jpeg_bytes = frame_file.read()


    # Convert bytes → NumPy array

    image_array = np.frombuffer(
        jpeg_bytes,
        dtype=np.uint8,
    )


    # Decode JPEG → OpenCV image

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )


    if image is None:

        return "could not decode frame", 400


    # Store frame for display

    update_latest_frame(

        image,

        device_id,

        timestamp_ms,

    )


    # Run your defect detection pipeline

    result = run_inference(

        image,

        device_id,

        timestamp_ms,

    )


    push_to_dashboard(result)


    return "", 200


# --------------------------------------------------
# Flask Server Thread
# --------------------------------------------------

def run_server():

    """
    Run Flask in a background thread.

    This allows OpenCV GUI to run on the main thread.
    """

    app.run(

        host="0.0.0.0",

        port=5000,

        threaded=True,

        use_reloader=False,

    )


# --------------------------------------------------
# OpenCV Display Loop
# --------------------------------------------------

def display_stream():

    print("[stream] Starting OpenCV display")


    # Windows are created lazily, one per device, the
    # first time we see a frame from that device.

    open_windows = set()


    while True:

        frames_to_display = []


        # Safely copy shared frame data

        with frame_lock:

            for device_id, data in latest_frames.items():

                frames_to_display.append(

                    (

                        device_id,

                        data["timestamp"],

                        data["image"].copy(),

                    )

                )


        if not frames_to_display:

            # Required to keep OpenCV GUI responsive

            key = cv2.waitKey(10) & 0xFF


            if key == ord("q"):

                break


            continue


        # --------------------------------------------------
        # Display every connected device
        # --------------------------------------------------

        for (

            device_id,

            timestamp_ms,

            image,

        ) in frames_to_display:


            # Spawn a dedicated OpenCV window the first
            # time this device is seen.

            if device_id not in open_windows:

                win = window_name_for(device_id)

                cv2.namedWindow(

                    win,

                    cv2.WINDOW_NORMAL,

                )

                cv2.resizeWindow(

                    win,

                    640,

                    480,

                )

                open_windows.add(device_id)

                fps_start_time[device_id] = time.time()

                fps_frame_count[device_id] = 0

                last_fps[device_id] = 0.0


            fps_frame_count[device_id] += 1


            elapsed = (

                time.time()

                - fps_start_time[device_id]

            )


            if elapsed >= FPS_WINDOW_SEC:

                last_fps[device_id] = (

                    fps_frame_count[device_id]

                    / elapsed

                )


                fps_frame_count[device_id] = 0

                fps_start_time[device_id] = time.time()


            # --------------------------------------------------
            # Overlay Information
            # --------------------------------------------------

            cv2.putText(

                image,

                f"Device: {device_id}",

                (10, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2,

            )


            cv2.putText(

                image,

                f"Timestamp: {timestamp_ms}",

                (10, 60),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 0),

                2,

            )


            cv2.putText(

                image,

                f"FPS: {last_fps[device_id]:.1f}",

                (10, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2,

            )


            # --------------------------------------------------
            # Show Frame
            # --------------------------------------------------

            cv2.imshow(

                window_name_for(device_id),

                image,

            )


        # --------------------------------------------------
        # GUI Event Processing
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF


        # Press Q to exit

        if key == ord("q"):

            break


    cv2.destroyAllWindows()


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # Flask runs in background

    server_thread = threading.Thread(

        target=run_server,

        daemon=True,

    )


    server_thread.start()


    # OpenCV GUI MUST run on main thread

    display_stream()