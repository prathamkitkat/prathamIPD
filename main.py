from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import json

# FastAPI is a backend framework,
# websockets enable real time streaming of data between client and server
# base64 is used to encode and decode images
# cv2 is used for image processing
# numpy is used for numerical operations
# json is used to format data

app = FastAPI() # created a FastAPI application where i will mount all the endpoints

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_base64_frame(data):
    img_bytes = base64.b64decode(data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

# this is a fastAPI decorator which is used to define a websocket endpoint
@app.websocket("/wsbicep")
async def websocket_bicep(ws: WebSocket):
    await ws.accept() # upgrades the connection from http to websocket

    from bicep_processor import ProcessFrameBicep
    # ProcessFrameBicep is a class that processes the frames for bicep curls
    from run_curl import pose # this is the mediapipe pose model

    processor = ProcessFrameBicep(flip_frame=True)
    # here i made an object of the class which will process each frame one by one
    while True:
        try:
            data = await ws.receive_text()

            frame = decode_base64_frame(data)
            processed_frame, form_issues = processor.process(frame, pose)
            encoded = encode_frame(processed_frame)

            stats = {
                "frame": encoded,
                "form_ok": len(form_issues) == 0,
                "feedback": form_issues,          # list of plain-text correction strings
                "left": {
                    "reps": processor.left_arm.counter,
                    "half_reps": processor.left_arm.half_reps,
                    "stage": processor.left_arm.stage,
                },
                "right": {
                    "reps": processor.right_arm.counter,
                    "half_reps": processor.right_arm.half_reps,
                    "stage": processor.right_arm.stage,
                },
            }

            await ws.send_text(json.dumps(stats))

        except Exception as e:
            print("🔥 ERROR:", e)
            import traceback
            traceback.print_exc()
            break


@app.websocket("/wssquat")
async def websocket_squat(ws: WebSocket):
    await ws.accept()

    from utils import get_mediapipe_pose
    from process_frame_squats import ProcessFrame
    from thresholds import get_thresholds_beginner

    thresholds = get_thresholds_beginner()
    pose = get_mediapipe_pose()
    processor = ProcessFrame(thresholds=thresholds, flip_frame=True)

    while True:
        try:
            data = await ws.receive_text()

            frame = decode_base64_frame(data)
            processed_frame, play_sound, feedback_msgs = processor.process(frame, pose)
            encoded = encode_frame(processed_frame)

            st = processor.state_tracker
            has_errors = bool(
                any(st['DISPLAY_TEXT']) or st['LOWER_HIPS']
            )

            stats = {
                "frame": encoded,
                "squat_count":   st['SQUAT_COUNT'],
                "improper_squat": st['IMPROPER_SQUAT'],
                "curr_state":    st['curr_state'],      # 's1' | 's2' | 's3' | None
                "form_ok":       not has_errors,
                "feedback":      feedback_msgs,
            }

            await ws.send_text(json.dumps(stats))

        except Exception as e:
            print("🔥 ERROR:", e)
            import traceback
            traceback.print_exc()
            break