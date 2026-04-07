from fastapi import FastAPI, WebSocket
import numpy as np
import cv2
import base64

# Import your existing modules

app = FastAPI()



def decode_base64_frame(data):
    img_bytes = base64.b64decode(data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


@app.websocket("/wsbicep")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    from utils import get_mediapipe_pose
    from process_frame_curling import ProcessFrame
    from threshold_curl import get_thresholds_beginner
    # Initialize once (IMPORTANT for performance)
    thresholds = get_thresholds_beginner()
    pose = get_mediapipe_pose()
    processor = ProcessFrame(thresholds=thresholds, flip_frame=True)

    while True:
        try:
            data = await ws.receive_text()
            # print("Frame received")

            frame = decode_base64_frame(data)

            processed_frame, _ = processor.process(frame, pose)

            encoded = encode_frame(processed_frame)

            await ws.send_text(encoded)

        except Exception as e:
            print("🔥 ERROR:", e)
            import traceback
            traceback.print_exc()
            break
        
@app.websocket("/wssquat")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    from utils import get_mediapipe_pose
    from process_frame_squats import ProcessFrame
    from thresholds import get_thresholds_beginner

    # Initialize once (IMPORTANT for performance)
    thresholds = get_thresholds_beginner()
    pose = get_mediapipe_pose()
    processor = ProcessFrame(thresholds=thresholds, flip_frame=True)

    while True:
        try:
            data = await ws.receive_text()
            # print("Frame received")

            frame = decode_base64_frame(data)

            processed_frame, _ = processor.process(frame, pose)

            encoded = encode_frame(processed_frame)

            await ws.send_text(encoded)

        except Exception as e:
            print("🔥 ERROR:", e)
            import traceback
            traceback.print_exc()
            break