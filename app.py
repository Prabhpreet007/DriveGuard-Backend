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
    app.run(host='0.0.0.0', port=5000, debug=False)