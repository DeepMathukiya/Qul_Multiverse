# DefectCam — Camera Streaming → ML Backend

This Android app runs on multiple phones and streams their camera frames over
HTTP to a single ML backend running on your laptop. The app does **not** run
any ML itself and does **not** display results — it's a one-way camera →
backend pipe. This doc is for whoever is writing the backend/ML side.

## What the app sends

Every ~700ms (configurable in-app), each phone POSTs one JPEG frame as
`multipart/form-data` to a URL you control:

```
POST http://<your-laptop-ip>:<port>/upload
Content-Type: multipart/form-data

  device_id   (text)   e.g. "Phone-1"   — set per phone in the app's settings bar
  timestamp   (text)   epoch millis, e.g. "1752233400123"
  frame       (file)   JPEG bytes, filename like "Phone-1_1752233400123.jpg"
```

- `device_id` is how you tell phones apart on the server/dashboard — it's a
  free-text label set once per phone in the app (defaults to something like
  `Pixel_7-a1b2`, but the user can rename it to `Phone-1` / `Phone-2` etc).
- The app ignores whatever your server responds with (status code, body —
  doesn't matter). Respond `200 OK` with an empty body; that's enough.
- Frames arrive from however many phones are running the app concurrently —
  your endpoint needs to handle interleaved requests from multiple
  `device_id`s, not just one stream.

## Network requirements

- Laptop and all phones must be on the **same Wi-Fi/LAN**.
- Bind your server to `0.0.0.0`, not `127.0.0.1`/`localhost` — otherwise
  phones on the network can't reach it.
- Open the port in your laptop's firewall (e.g. `sudo ufw allow 5000`).
- Get the laptop's LAN IP with `hostname -I` (Linux) or `ipconfig` (Windows),
  and enter `http://<that-ip>:<port>/upload` into each phone's settings bar
  in the app (no rebuild needed — it's a runtime setting).

## Minimal example server

A working, runnable example is in [`server/example_receiver.py`](server/example_receiver.py).
It receives frames, decodes them to a NumPy/OpenCV image, and has a clearly
marked spot to plug in your model. Run it with:

```bash
cd server
pip install flask opencv-python numpy
python example_receiver.py
```

Then point the app's server URL at `http://<laptop-ip>:5000/upload`.

Quick manual test without a phone, from the laptop itself:

```bash
curl -F "device_id=test" -F "timestamp=123" -F "frame=@some_photo.jpg" \
  http://localhost:5000/upload
```

## Wiring in your model

Inside the endpoint, after decoding `frame` into an image:

```python
result = your_model.predict(image)   # e.g. dent / no-dent, bounding box, confidence
```

What you do with `result` is up to you — the app never sees it. Typical
next step for the "redirect into dashboard" part of the pipeline:
- push `{device_id, timestamp, result}` to a dashboard over a WebSocket / SSE
  connection, or
- write it to a database/queue that the dashboard polls, or
- broadcast over `Flask-SocketIO` to a browser dashboard.

## Multiple phones at a glance

| Phone   | In-app device_id | In-app server URL                  |
|---------|-------------------|-------------------------------------|
| Phone A | `Phone-1`          | `http://192.168.1.50:5000/upload` |
| Phone B | `Phone-2`          | `http://192.168.1.50:5000/upload` |

Both phones point at the **same** URL — the `device_id` field is what lets
the server (and dashboard) distinguish which phone a frame came from.
