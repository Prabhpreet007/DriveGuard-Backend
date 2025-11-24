# import base64
# import math
# import time
# import logging

# import cv2
# import mediapipe as mp
# import numpy as np
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # change in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ========= MediaPipe Init =========
# try:
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5,
#     )
#     logger.info("✅ MediaPipe FaceMesh initialized")
# except Exception as e:
#     logger.error(f"❌ MediaPipe init failed: {e}")
#     face_mesh = None

# # ========= Global State =========
# alert_state = {
#     "alert_active": False,
#     "alert_type": "",
#     "eye_closed_frames": 0,
#     "yawn_frames": 0,
#     "tilt_frames": 0,
#     "last_alert_time": 0,
#     "sound_played": False,
# }

# DEFAULT_THRESHOLDS = {
#     "ear_threshold": 0.22,
#     "mar_threshold": 0.40,
#     "tilt_threshold": 15,
# }

# # ========= Helper functions =========
# from scipy.spatial import distance

# def eye_aspect_ratio(eye):
#     try:
#         A = distance.euclidean(eye[1], eye[5])
#         B = distance.euclidean(eye[2], eye[4])
#         C = distance.euclidean(eye[0], eye[3])
#         ear = (A + B) / (2.0 * C)
#         return ear
#     except Exception:
#         return 0.5

# def mouth_aspect_ratio(mouth):
#     try:
#         A = distance.euclidean(mouth[0], mouth[1])  # 13,14
#         B = distance.euclidean(mouth[2], mouth[3])  # 78,308
#         mar = A / B
#         return mar
#     except Exception:
#         return 0.3

# def calculate_tilt(landmarks, img_shape):
#     try:
#         h, w = img_shape[:2]
#         forehead = landmarks[10]
#         chin = landmarks[152]
#         dx = chin[0] - forehead[0]
#         dy = chin[1] - forehead[1]
#         angle = math.degrees(math.atan2(dy, dx))
#         tilt = abs(angle - 90) if angle > 90 else abs(90 - angle)
#         return tilt
#     except Exception:
#         return 0.0

# def process_frame(frame_data, thresholds):
#     global alert_state

#     try:
#         if ',' in frame_data:
#             frame_data = frame_data.split(',')[1]

#         img_data = base64.b64decode(frame_data)
#         np_arr = np.frombuffer(img_data, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         if img is None:
#             return {"error": "Invalid image"}, "Image decoding failed"

#         # Resize to speed up
#         img = cv2.resize(img, (320, 240))

#         rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         if face_mesh is None:
#             return {"error": "MediaPipe not initialized"}, "Face mesh not available"

#         results = face_mesh.process(rgb_frame)
#         h, w = img.shape[:2]

#         current_time = time.time()
#         if current_time - alert_state["last_alert_time"] > 5:
#             alert_state["sound_played"] = False

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

#                 left_eye_idx = [33, 160, 158, 133, 153, 144]
#                 right_eye_idx = [362, 385, 387, 263, 373, 380]
#                 mouth_idx = [13, 14, 78, 308]

#                 left_eye = landmarks[left_eye_idx]
#                 right_eye = landmarks[right_eye_idx]
#                 mouth = landmarks[mouth_idx]

#                 ear_left = eye_aspect_ratio(left_eye)
#                 ear_right = eye_aspect_ratio(right_eye)
#                 ear = (ear_left + ear_right) / 2.0
#                 mar = mouth_aspect_ratio(mouth)
#                 tilt = calculate_tilt(landmarks, img.shape)

#                 alert_detected = False
#                 alert_type = ""

#                 # Eye
#                 if ear < thresholds["ear_threshold"]:
#                     alert_state["eye_closed_frames"] += 1
#                     if alert_state["eye_closed_frames"] >= 30:
#                         alert_detected = True
#                         alert_type = "drowsiness"
#                 else:
#                     alert_state["eye_closed_frames"] = 0

#                 # Yawn
#                 if mar > thresholds["mar_threshold"] and not alert_detected:
#                     alert_state["yawn_frames"] += 1
#                     if alert_state["yawn_frames"] >= 20:
#                         alert_detected = True
#                         alert_type = "yawning"
#                 else:
#                     alert_state["yawn_frames"] = 0

#                 # Tilt
#                 if tilt > thresholds["tilt_threshold"] and not alert_detected:
#                     alert_state["tilt_frames"] += 1
#                     if alert_state["tilt_frames"] >= 30:
#                         alert_detected = True
#                         alert_type = "tilt"
#                 else:
#                     alert_state["tilt_frames"] = 0

#                 if alert_detected:
#                     if (not alert_state["alert_active"] or
#                         current_time - alert_state["last_alert_time"] > 5):
#                         alert_state["alert_active"] = True
#                         alert_state["alert_type"] = alert_type
#                         alert_state["last_alert_time"] = current_time
#                         alert_state["sound_played"] = False
#                 else:
#                     if alert_state["alert_active"] and current_time - alert_state["last_alert_time"] > 2:
#                         alert_state["alert_active"] = False
#                         alert_state["alert_type"] = ""

#                 alert_info = {
#                     "alert": alert_state["alert_active"],
#                     "type": alert_state["alert_type"] if alert_state["alert_active"] else "",
#                     "metrics": {
#                         "ear": round(ear, 3),
#                         "mar": round(mar, 3),
#                         "tilt": round(tilt, 1),
#                         "eye_frames": alert_state["eye_closed_frames"],
#                         "yawn_frames": alert_state["yawn_frames"],
#                         "tilt_frames": alert_state["tilt_frames"]
#                     },
#                     "status": "ALERT!" if alert_state["alert_active"] else "NORMAL",
#                     "sound_alert": alert_state["alert_active"] and not alert_state["sound_played"],
#                     "detector_type": "mediapipe"
#                 }

#                 if alert_info["sound_alert"]:
#                     alert_state["sound_played"] = True

#                 return alert_info, None

#         else:
#             alert_info = {
#                 "alert": False,
#                 "type": "",
#                 "metrics": {
#                     "ear": 0,
#                     "mar": 0,
#                     "tilt": 0,
#                     "eye_frames": 0,
#                     "yawn_frames": 0,
#                     "tilt_frames": 0
#                 },
#                 "status": "NO FACE DETECTED",
#                 "sound_alert": False,
#                 "detector_type": "mediapipe"
#             }
#             return alert_info, None

#     except Exception as e:
#         logger.error(f"Process frame error: {e}")
#         return {"error": str(e)}, "Processing failed"


# # ========= REST health =========
# @app.get("/api/health")
# async def health():
#     return {
#         "status": "healthy",
#         "mediapipe": "loaded" if face_mesh else "failed"
#     }


# # ========= WebSocket Endpoint =========
# @app.websocket("/ws/drowsiness")
# async def drowsiness_ws(websocket: WebSocket):
#     await websocket.accept()
#     logger.info("✅ WebSocket client connected")
#     try:
#         while True:
#             data = await websocket.receive_json()
#             frame_data = data.get("frame")
#             thresholds = data.get("thresholds", DEFAULT_THRESHOLDS)

#             if not frame_data:
#                 await websocket.send_json({"error": "No frame data"})
#                 continue

#             result, error = process_frame(frame_data, thresholds)
#             if error:
#                 await websocket.send_json({"error": error})
#             else:
#                 await websocket.send_json(result)

#     except WebSocketDisconnect:
#         logger.info("❌ WebSocket client disconnected")
#     except Exception as e:
#         logger.error(f"WebSocket error: {e}")

# # Run with: uvicorn main:app --host 0.0.0.0 --port 8000
