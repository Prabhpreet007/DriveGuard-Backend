# import streamlit as st
# import cv2
# import base64
# import mediapipe as mp
# import numpy as np
# import math
# import time
# from scipy.spatial import distance
# import av
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
# import streamlit.components.v1 as components

# st.set_page_config(page_title="DriveGuard Web", layout="centered")

# st.markdown(
#     """
#     <h2 style='text-align:center;color:#00BFFF;'>🚗 DriveGuard Web - Drowsiness Detection</h2>
#     """, unsafe_allow_html=True
# )

# mp_face = mp.solutions.face_mesh

# st.sidebar.header("⚙️ Threshold Settings")
# thresh = st.sidebar.slider("Eye Aspect Ratio (Drowsiness Threshold)", 0.15, 0.35, 0.22, 0.01)
# mar_thresh = st.sidebar.slider("Mouth Aspect Ratio (Yawn Threshold)", 0.4, 0.8, 0.40, 0.01)
# tilt_thresh = st.sidebar.slider("Tilt Angle Threshold (degrees)", 10, 30, 15, 1)

# st.sidebar.header("🔊 Alert Settings")
# enable_sound = st.sidebar.checkbox("Enable Sound Alerts", value=True)

# # Load sound file and get base64
# def load_audio_base64(audio_file_path):
#     try:
#         with open(audio_file_path, "rb") as f:
#             audio_bytes = f.read()
#         return base64.b64encode(audio_bytes).decode()
#     except Exception as e:
#         st.error(f"Audio file load error: {e}")
#         return None

# alert_sound_b64 = load_audio_base64("music.wav")

# # Inject invisible audio element ONCE after Test Sound button is clicked
# def inject_audio_element(file_b64):
#     components.html(f"""
#         <audio id="alert_player" src="data:audio/wav;base64,{file_b64}" preload="auto"></audio>
#         <script>
#             window.alertPlayer = document.getElementById('alert_player');
#         </script>
#     """, height=0)

# # Play audio by JS when alert
# def play_audio_js():
#     components.html("""
#         <script>
#             if(window.alertPlayer){
#                 window.alertPlayer.currentTime = 0;
#                 window.alertPlayer.play();
#             }
#         </script>
#     """, height=0)

# def eye_aspect_ratio(eye):
#     try:
#         A = distance.euclidean(eye[1], eye[5])
#         B = distance.euclidean(eye[2], eye[4])
#         C = distance.euclidean(eye[0], eye[3])
#         ear = (A + B) / (2.0 * C)
#         return ear
#     except:
#         return 0.5

# def mouth_aspect_ratio(mouth):
#     try:
#         A = distance.euclidean(mouth[13], mouth[14])
#         B = distance.euclidean(mouth[78], mouth[308])
#         mar = A / B
#         return mar
#     except:
#         return 0.3

# def calculate_tilt(face_landmarks, img_shape):
#     try:
#         h, w = img_shape[:2]
#         forehead = face_landmarks[10]
#         chin = face_landmarks[152]
#         dx = chin[0] - forehead[0]
#         dy = chin[1] - forehead[1]
#         angle = math.degrees(math.atan2(dy, dx))
#         tilt = abs(angle - 90) if angle > 90 else abs(90 - angle)
#         return tilt
#     except:
#         return 0.0

# class VideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.face_mesh = mp_face.FaceMesh(
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.7
#         )
#         self.alert_active = False
#         self.last_alert_type = ""
#         self.eye_closed_frames = 0
#         self.yawn_frames = 0
#         self.tilt_frames = 0
#         self.sound_played = False
#         self.last_sound_time = 0
#         self.frame_count = 0
        
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         img = cv2.flip(img, 1)
#         rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = self.face_mesh.process(rgb_frame)
#         h, w, _ = img.shape

#         self.frame_count += 1
#         alert_text = ""
#         alert_color = (0, 0, 255)
#         current_time = time.time()
        
#         if current_time - self.last_sound_time > 5:
#             self.sound_played = False
        
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

#                 left_eye_idx = [33, 160, 158, 133, 153, 144]
#                 right_eye_idx = [362, 385, 387, 263, 373, 380]
#                 mouth_idx = [13, 14, 78, 308, 80, 88, 87, 317, 318, 324, 402]

#                 left_eye = landmarks[left_eye_idx]
#                 right_eye = landmarks[right_eye_idx]
#                 mouth = landmarks[mouth_idx]

#                 ear_left = eye_aspect_ratio(left_eye)
#                 ear_right = eye_aspect_ratio(right_eye)
#                 ear = (ear_left + ear_right) / 2.0
                
#                 mar = mouth_aspect_ratio(mouth)
#                 tilt = calculate_tilt(landmarks, img.shape)

#                 self.alert_active = False
#                 alert_text = ""

#                 if ear < thresh:
#                     self.eye_closed_frames += 1
#                     if self.eye_closed_frames >= 90:
#                         alert_text = "😴 DROWSINESS DETECTED!"
#                         self.alert_active = True
#                         self.last_alert_type = "drowsiness"
#                         if enable_sound and not self.sound_played:
#                             self.sound_played = True
#                             self.last_sound_time = current_time
#                 else:
#                     self.eye_closed_frames = 0

#                 if mar > mar_thresh and not self.alert_active:
#                     self.yawn_frames += 1
#                     if self.yawn_frames >= 60:
#                         alert_text = "😮 YAWNING DETECTED!"
#                         self.alert_active = True
#                         self.last_alert_type = "yawning"
#                         if enable_sound and not self.sound_played:
#                             self.sound_played = True
#                             self.last_sound_time = current_time
#                 else:
#                     self.yawn_frames = 0

#                 if tilt > tilt_thresh and not self.alert_active:
#                     self.tilt_frames += 1
#                     if self.tilt_frames >= 90:
#                         alert_text = "↔️ HEAD TILT ALERT!"
#                         self.alert_active = True
#                         self.last_alert_type = "tilt"
#                         if enable_sound and not self.sound_played:
#                             self.sound_played = True
#                             self.last_sound_time = current_time
#                 else:
#                     self.tilt_frames = 0

#                 ear_color = (0, 255, 0) if ear > thresh else (0, 0, 255)
#                 mar_color = (0, 255, 0) if mar < mar_thresh else (0, 0, 255)
#                 tilt_color = (0, 255, 0) if tilt < tilt_thresh else (0, 0, 255)
                
#                 cv2.putText(img, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
#                 cv2.putText(img, f"MAR: {mar:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)
#                 cv2.putText(img, f"TILT: {tilt:.1f}°", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tilt_color, 2)
#                 cv2.putText(img, f"Eye Frames: {self.eye_closed_frames}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
#                 cv2.putText(img, f"Yawn Frames: {self.yawn_frames}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
#                 cv2.putText(img, f"Tilt Frames: {self.tilt_frames}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
#                 status_color = (0, 255, 0)
#                 status_text = "NORMAL"
#                 if self.alert_active:
#                     status_color = (0, 0, 255)
#                     status_text = "ALERT!"
                
#                 cv2.putText(img, f"STATUS: {status_text}", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
#                 for idx in left_eye_idx + right_eye_idx:
#                     x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
#                     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                
#                 for idx in mouth_idx:
#                     x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
#                     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
                
#                 if alert_text:
#                     cv2.putText(img, alert_text, (w//2 - 150, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
#                     cv2.rectangle(img, (5, 5), (w-5, h-5), alert_color, 5)
#         else:
#             cv2.putText(img, "NO FACE DETECTED", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#             self.alert_active = False
#             self.eye_closed_frames = 0
#             self.yawn_frames = 0
#             self.tilt_frames = 0

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# def main():
#     st.info("🎥 Real-time Driver Monitoring - Detects drowsiness, yawning, and head tilt")

#     alert_placeholder = st.empty()
#     sound_placeholder = st.empty()

#     if alert_sound_b64 is None:
#         st.error("❌ music.wav file not found! Please make sure music.wav is in the same folder.")
#         st.info("Place your music.wav file in the same directory as this script.")

#     # Permission: inject audio element once after test sound
#     if st.sidebar.button("🔊 Test Sound"):
#         st.session_state['audio_permission'] = True
#         inject_audio_element(alert_sound_b64)

#     ctx = webrtc_streamer(
#         key="driver-monitoring",
#         mode=WebRtcMode.SENDRECV,
#         video_processor_factory=VideoProcessor,
#         media_stream_constraints={
#             "video": {
#                 "width": {"ideal": 480},
#                 "height": {"ideal": 360}
#             },
#             "audio": True
#         },
#         async_processing=True,
#     )

#     with st.expander("📋 Instructions & Info"):
#         st.markdown("""
#         **How to use:**
#         1. **Allow both camera and microphone access** when prompted
#         2. Sit straight and look at camera
#         3. System will alert after continuous detection
        
#         **Detection Timing:**
#         - 🚨 **Drowsiness**: Eyes closed for 3+ seconds
#         - 🚨 **Yawning**: Mouth open wide for 2+ seconds
#         - 🚨 **Head Tilt**: Head tilted for 3+ seconds
        
#         **Current Thresholds:**
#         - EAR < {:.2f} (Drowsiness)
#         - MAR > {:.2f} (Yawning)
#         - Tilt > {}° (Head Tilt)
#         """.format(thresh, mar_thresh, tilt_thresh))
    
#     # Trigger sound when alert if permission
#     if ctx.video_processor:
#         if ctx.video_processor.alert_active:
#             alert_placeholder.error(f"🚨 **ALERT**: {ctx.video_processor.last_alert_type.upper()} DETECTED!")
#             if (
#                 enable_sound and
#                 'audio_permission' in st.session_state and
#                 st.session_state['audio_permission'] and
#                 ctx.video_processor.sound_played and
#                 alert_sound_b64 is not None
#             ):
#                 play_audio_js()
#                 ctx.video_processor.sound_played = False
#         else:
#             alert_placeholder.success("✅ **Status**: Normal - All parameters safe")
#             sound_placeholder.empty()

# if __name__ == "__main__":
#     main()




































import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import math
from scipy.spatial import distance
import base64
import time

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Global variables for state management
alert_state = {
    "alert_active": False,
    "alert_type": "",
    "eye_closed_frames": 0,
    "yawn_frames": 0,
    "tilt_frames": 0,
    "last_alert_time": 0
}

# Default thresholds
DEFAULT_THRESHOLDS = {
    "ear_threshold": 0.22,
    "mar_threshold": 0.40,
    "tilt_threshold": 15
}

def eye_aspect_ratio(eye):
    try:
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    except:
        return 0.5

def mouth_aspect_ratio(mouth):
    try:
        A = distance.euclidean(mouth[13], mouth[14])
        B = distance.euclidean(mouth[78], mouth[308])
        mar = A / B
        return mar
    except:
        return 0.3

def calculate_tilt(face_landmarks, img_shape):
    try:
        h, w = img_shape[:2]
        forehead = face_landmarks[10]
        chin = face_landmarks[152]
        dx = chin[0] - forehead[0]
        dy = chin[1] - forehead[1]
        angle = math.degrees(math.atan2(dy, dx))
        tilt = abs(angle - 90) if angle > 90 else abs(90 - angle)
        return tilt
    except:
        return 0.0

def process_frame(frame_data, thresholds):
    global alert_state
    
    # Decode base64 image
    img_data = base64.b64decode(frame_data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, "Invalid image"
    
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h, w = img.shape[:2]
    
    current_time = time.time()
    alert_info = {
        "alert": False,
        "type": "",
        "metrics": {},
        "frame_data": None
    }
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

            # Landmark indices
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            mouth_idx = [13, 14, 78, 308]

            left_eye = landmarks[left_eye_idx]
            right_eye = landmarks[right_eye_idx]
            mouth = landmarks[mouth_idx]

            # Calculate metrics
            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)
            ear = (ear_left + ear_right) / 2.0
            mar = mouth_aspect_ratio(mouth)
            tilt = calculate_tilt(landmarks, img.shape)

            # Check alerts
            alert_detected = False
            alert_type = ""
            
            # Drowsiness detection (eyes closed)
            if ear < thresholds["ear_threshold"]:
                alert_state["eye_closed_frames"] += 1
                if alert_state["eye_closed_frames"] >= 90:  # ~3 seconds
                    alert_detected = True
                    alert_type = "drowsiness"
            else:
                alert_state["eye_closed_frames"] = 0

            # Yawning detection
            if mar > thresholds["mar_threshold"] and not alert_detected:
                alert_state["yawn_frames"] += 1
                if alert_state["yawn_frames"] >= 60:  # ~2 seconds
                    alert_detected = True
                    alert_type = "yawning"
            else:
                alert_state["yawn_frames"] = 0

            # Head tilt detection
            if tilt > thresholds["tilt_threshold"] and not alert_detected:
                alert_state["tilt_frames"] += 1
                if alert_state["tilt_frames"] >= 90:  # ~3 seconds
                    alert_detected = True
                    alert_type = "tilt"
            else:
                alert_state["tilt_frames"] = 0

            # Update alert state
            if alert_detected and (current_time - alert_state["last_alert_time"] > 5):
                alert_state["alert_active"] = True
                alert_state["alert_type"] = alert_type
                alert_state["last_alert_time"] = current_time
            elif current_time - alert_state["last_alert_time"] > 5:
                alert_state["alert_active"] = False

            # Prepare response
            alert_info = {
                "alert": alert_state["alert_active"],
                "type": alert_state["alert_type"] if alert_state["alert_active"] else "",
                "metrics": {
                    "ear": round(ear, 3),
                    "mar": round(mar, 3),
                    "tilt": round(tilt, 1),
                    "eye_frames": alert_state["eye_closed_frames"],
                    "yawn_frames": alert_state["yawn_frames"],
                    "tilt_frames": alert_state["tilt_frames"]
                },
                "status": "ALERT!" if alert_state["alert_active"] else "NORMAL"
            }
            
            # Encode processed image for display
            _, buffer = cv2.imencode('.jpg', img)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            alert_info["frame_data"] = f"data:image/jpeg;base64,{frame_base64}"
            
            break
    else:
        alert_info = {
            "alert": False,
            "type": "",
            "metrics": {
                "ear": 0,
                "mar": 0,
                "tilt": 0,
                "eye_frames": 0,
                "yawn_frames": 0,
                "tilt_frames": 0
            },
            "status": "NO FACE DETECTED",
            "frame_data": None
        }
    
    return alert_info, None

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Driver Drowsiness API is running"})

@app.route('/api/process-frame', methods=['POST'])
def process_frame_route():
    try:
        data = request.json
        frame_data = data.get('frame')
        thresholds = data.get('thresholds', DEFAULT_THRESHOLDS)
        
        if not frame_data:
            return jsonify({"error": "No frame data provided"}), 400
        
        result, error = process_frame(frame_data, thresholds)
        
        if error:
            return jsonify({"error": error}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset-alerts', methods=['POST'])
def reset_alerts():
    global alert_state
    alert_state = {
        "alert_active": False,
        "alert_type": "",
        "eye_closed_frames": 0,
        "yawn_frames": 0,
        "tilt_frames": 0,
        "last_alert_time": 0
    }
    return jsonify({"message": "Alerts reset successfully"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render ke liye
    app.run(host='0.0.0.0', port=port, debug=False)






































# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# import math
# from scipy.spatial import distance
# import base64
# import time
# import logging
# import os

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)

# class DrowsinessDetector:
#     def __init__(self):
#         try:
#             # Load OpenCV Haar cascades
#             self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#             self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#             logger.info("✅ OpenCV cascades loaded successfully")
#             self.detector_type = "opencv"
#         except Exception as e:
#             logger.error(f"❌ Cascade loading failed: {e}")
#             raise

#     def detect_face_features(self, gray_frame):
#         """Detect face and eyes using OpenCV"""
#         faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
#         if len(faces) == 0:
#             return None
            
#         x, y, w, h = faces[0]
        
#         # Get face region
#         face_roi = gray_frame[y:y+h, x:x+w]
        
#         # Detect eyes in the face region
#         eyes = self.eye_cascade.detectMultiScale(face_roi)
        
#         # Calculate basic metrics
#         eye_states = []
#         for (ex, ey, ew, eh) in eyes:
#             eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
#             # Simple eye open/close detection based on intensity
#             eye_openness = np.mean(eye_roi) / 255.0
#             eye_states.append(eye_openness)
        
#         # Simulate face landmarks for compatibility
#         landmarks = self._simulate_landmarks(x, y, w, h, len(eyes) > 0)
        
#         return {
#             'landmarks': landmarks,
#             'face_rect': (x, y, w, h),
#             'eyes_detected': len(eyes),
#             'eye_states': eye_states
#         }
    
#     def _simulate_landmarks(self, x, y, w, h, eyes_visible):
#         """Simulate face landmarks for calculations"""
#         landmarks = []
        
#         # Create 68 points like dlib (for compatibility)
#         for i in range(68):
#             if i in [36, 37, 38, 39, 40, 41]:  # Left eye
#                 landmarks.append((x + w//4 + (i-36)*5, y + h//3))
#             elif i in [42, 43, 44, 45, 46, 47]:  # Right eye
#                 landmarks.append((x + 3*w//4 + (i-42)*5, y + h//3))
#             elif i in [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]:  # Mouth outer
#                 landmarks.append((x + w//2 + (i-54)*3, y + 2*h//3))
#             elif i in [60, 61, 62, 63, 64, 65, 66, 67]:  # Mouth inner
#                 landmarks.append((x + w//2 + (i-63)*2, y + 2*h//3 + 10))
#             elif i == 30:  # Nose tip
#                 landmarks.append((x + w//2, y + h//2))
#             elif i == 8:   # Chin
#                 landmarks.append((x + w//2, y + h))
#             elif i == 27:  # Nose bridge
#                 landmarks.append((x + w//2, y + h//3))
#             elif i in [17, 18, 19, 20, 21]:  # Right eyebrow
#                 landmarks.append((x + 3*w//4 + (i-17)*5, y + h//4))
#             elif i in [22, 23, 24, 25, 26]:  # Left eyebrow
#                 landmarks.append((x + w//4 + (i-22)*5, y + h//4))
#             else:
#                 # Fill other points
#                 landmarks.append((x + (i % w), y + (i % h)))
        
#         return landmarks

# # Initialize detector
# detector = DrowsinessDetector()

# # Global state
# alert_state = {
#     "alert_active": False,
#     "alert_type": "",
#     "eye_closed_frames": 0,
#     "yawn_frames": 0,
#     "tilt_frames": 0,
#     "last_alert_time": 0
# }

# DEFAULT_THRESHOLDS = {
#     "ear_threshold": 0.25,
#     "mar_threshold": 0.52,
#     "tilt_threshold": 15
# }

# def eye_aspect_ratio(eye):
#     """Calculate Eye Aspect Ratio"""
#     try:
#         if len(eye) < 6:
#             return 0.5
            
#         # Use dlib-like indices: [36, 37, 38, 39, 40, 41] for left eye
#         A = distance.euclidean(eye[1], eye[5])  # Vertical 1
#         B = distance.euclidean(eye[2], eye[4])  # Vertical 2  
#         C = distance.euclidean(eye[0], eye[3])  # Horizontal
#         ear = (A + B) / (2.0 * C)
#         return max(0.1, min(ear, 0.5))
#     except Exception as e:
#         logger.warning(f"EAR calculation error: {e}")
#         return 0.3

# def mouth_aspect_ratio(mouth):
#     """Calculate Mouth Aspect Ratio"""
#     try:
#         if len(mouth) < 12:
#             return 0.3
            
#         # Use mouth outer points [48, 54, 51, 57] for MAR calculation
#         A = distance.euclidean(mouth[3], mouth[9])   # Vertical 1
#         B = distance.euclidean(mouth[2], mouth[10])  # Vertical 2
#         C = distance.euclidean(mouth[4], mouth[8])   # Vertical 3
#         D = distance.euclidean(mouth[0], mouth[6])   # Horizontal
        
#         mar = (A + B + C) / (3.0 * D)
#         return max(0.2, min(mar, 1.0))
#     except Exception as e:
#         logger.warning(f"MAR calculation error: {e}")
#         return 0.3

# def calculate_tilt(landmarks):
#     """Calculate head tilt angles"""
#     try:
#         if len(landmarks) < 68:
#             return 0, 0
            
#         # Get key points
#         left_eye = np.mean([landmarks[i] for i in [36, 37, 38, 39, 40, 41]], axis=0)
#         right_eye = np.mean([landmarks[i] for i in [42, 43, 44, 45, 46, 47]], axis=0)
#         nose_tip = landmarks[30]
        
#         # Calculate horizontal tilt (head sideways)
#         eye_dx = right_eye[0] - left_eye[0]
#         eye_dy = right_eye[1] - left_eye[1]
#         h_angle = math.degrees(math.atan2(eye_dy, eye_dx))
        
#         # Calculate vertical tilt (head up/down)
#         eyes_mid = (left_eye + right_eye) / 2
#         vertical_dx = nose_tip[0] - eyes_mid[0]
#         vertical_dy = nose_tip[1] - eyes_mid[1]
#         v_angle = math.degrees(math.atan2(vertical_dx, vertical_dy))
        
#         return abs(h_angle), abs(v_angle)
#     except Exception as e:
#         logger.warning(f"Tilt calculation error: {e}")
#         return 0, 0

# def process_frame(frame_data, thresholds):
#     global alert_state
    
#     try:
#         # Decode base64 image
#         if ',' in frame_data:
#             frame_data = frame_data.split(',')[1]
        
#         img_data = base64.b64decode(frame_data)
#         np_arr = np.frombuffer(img_data, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
#         if img is None:
#             return {"error": "Invalid image data"}, "Image decoding failed"
        
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         h, w = img.shape[:2]
        
#         current_time = time.time()
        
#         # Detect face features
#         features = detector.detect_face_features(gray)
        
#         alert_info = {
#             "alert": False,
#             "type": "",
#             "metrics": {},
#             "status": "NO FACE DETECTED",
#             "detector_type": detector.detector_type,
#             "frame_data": None
#         }
        
#         if features:
#             landmarks = features['landmarks']
#             face_rect = features['face_rect']
            
#             # Extract points for calculations
#             left_eye = [landmarks[i] for i in [36, 37, 38, 39, 40, 41]]
#             right_eye = [landmarks[i] for i in [42, 43, 44, 45, 46, 47]]
#             mouth = [landmarks[i] for i in range(48, 68)]  # All mouth points

#             # Calculate metrics
#             ear_left = eye_aspect_ratio(left_eye)
#             ear_right = eye_aspect_ratio(right_eye)
#             ear = (ear_left + ear_right) / 2.0
#             mar = mouth_aspect_ratio(mouth)
#             h_tilt, v_tilt = calculate_tilt(landmarks)
#             tilt = max(h_tilt, v_tilt)  # Use maximum tilt

#             # Adjust metrics based on actual eye detection
#             if features['eyes_detected'] == 0:
#                 ear = 0.1  # Eyes closed
#             elif features['eyes_detected'] > 0 and features['eye_states']:
#                 avg_eye_state = np.mean(features['eye_states'])
#                 ear = max(0.1, avg_eye_state * 0.4)

#             # Alert detection logic
#             alert_detected = False
#             alert_type = ""
            
#             # Drowsiness detection
#             if ear < thresholds["ear_threshold"]:
#                 alert_state["eye_closed_frames"] += 1
#                 if alert_state["eye_closed_frames"] >= 20:  # ~1 second
#                     alert_detected = True
#                     alert_type = "drowsiness"
#             else:
#                 alert_state["eye_closed_frames"] = 0

#             # Yawn detection
#             if mar > thresholds["mar_threshold"] and not alert_detected:
#                 alert_state["yawn_frames"] += 1
#                 if alert_state["yawn_frames"] >= 15:  # ~0.75 seconds
#                     alert_detected = True
#                     alert_type = "yawning"
#             else:
#                 alert_state["yawn_frames"] = 0

#             # Head tilt detection
#             if tilt > thresholds["tilt_threshold"] and not alert_detected:
#                 alert_state["tilt_frames"] += 1
#                 if alert_state["tilt_frames"] >= 25:  # ~1.25 seconds
#                     alert_detected = True
#                     alert_type = "tilt"
#             else:
#                 alert_state["tilt_frames"] = 0

#             # Update alert state
#             if alert_detected and (current_time - alert_state["last_alert_time"] > 5):
#                 alert_state["alert_active"] = True
#                 alert_state["alert_type"] = alert_type
#                 alert_state["last_alert_time"] = current_time
#             elif current_time - alert_state["last_alert_time"] > 5:
#                 alert_state["alert_active"] = False

#             # Prepare response
#             alert_info = {
#                 "alert": alert_state["alert_active"],
#                 "type": alert_state["alert_type"] if alert_state["alert_active"] else "",
#                 "metrics": {
#                     "ear": round(ear, 3),
#                     "mar": round(mar, 3),
#                     "tilt": round(tilt, 1),
#                     "eye_frames": alert_state["eye_closed_frames"],
#                     "yawn_frames": alert_state["yawn_frames"],
#                     "tilt_frames": alert_state["tilt_frames"],
#                     "eyes_detected": features['eyes_detected']
#                 },
#                 "status": "ALERT!" if alert_state["alert_active"] else "NORMAL",
#                 "detector_type": detector.detector_type
#             }
            
#             # Draw visualization
#             x, y, w, h = face_rect
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
#             # Draw eyes
#             for eye_idx in [[36, 37, 38, 39, 40, 41], [42, 43, 44, 45, 46, 47]]:
#                 eye_points = np.array([landmarks[i] for i in eye_idx], np.int32)
#                 cv2.polylines(img, [eye_points], True, (0, 255, 0), 1)
            
#             # Draw mouth
#             mouth_points = np.array([landmarks[i] for i in range(48, 60)], np.int32)
#             cv2.polylines(img, [mouth_points], True, (0, 165, 255), 1)
            
#             # Add metrics text
#             cv2.putText(img, f"EAR: {ear:.2f}", (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
#                        (0, 0, 255) if ear < thresholds["ear_threshold"] else (0, 255, 0), 2)
#             cv2.putText(img, f"MAR: {mar:.2f}", (10, 60), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
#                        (0, 0, 255) if mar > thresholds["mar_threshold"] else (0, 255, 0), 2)
#             cv2.putText(img, f"TILT: {tilt:.1f}°", (10, 90), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
#                        (0, 0, 255) if tilt > thresholds["tilt_threshold"] else (0, 255, 0), 2)
            
#             status_color = (0, 0, 255) if alert_info['alert'] else (0, 255, 0)
#             cv2.putText(img, f"Status: {alert_info['status']}", (10, 120), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
#             # Encode processed image
#             _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
#             frame_base64 = base64.b64encode(buffer).decode('utf-8')
#             alert_info["frame_data"] = f"data:image/jpeg;base64,{frame_base64}"
            
#         return alert_info, None
        
#     except Exception as e:
#         logger.error(f"Frame processing error: {e}")
#         return {"error": str(e)}, "Processing failed"

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({
#         "status": "healthy", 
#         "message": "DriveGuard API is running",
#         "detector_type": detector.detector_type
#     })

# @app.route('/api/process-frame', methods=['POST'])
# def process_frame_route():
#     try:
#         data = request.json
#         frame_data = data.get('frame')
#         thresholds = data.get('thresholds', DEFAULT_THRESHOLDS)
        
#         if not frame_data:
#             return jsonify({"error": "No frame data provided"}), 400
        
#         result, error = process_frame(frame_data, thresholds)
        
#         if error:
#             return jsonify({"error": error}), 400
        
#         return jsonify(result)
        
#     except Exception as e:
#         logger.error(f"API error: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/api/reset-alerts', methods=['POST'])
# def reset_alerts():
#     global alert_state
#     alert_state = {
#         "alert_active": False,
#         "alert_type": "",
#         "eye_closed_frames": 0,
#         "yawn_frames": 0,
#         "tilt_frames": 0,
#         "last_alert_time": 0
#     }
#     return jsonify({"message": "Alerts reset successfully"})

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=False)