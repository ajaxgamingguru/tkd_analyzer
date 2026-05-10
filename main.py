from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from typing import Dict, Any

app = FastAPI(title="TKD Technique Analyzer")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class TaekwondoAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def calculate_angle(self, a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def process_video(self, video_path: str, technique: str = "side_kick") -> Dict:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        output_path = video_path.replace(".mp4", "_overlay.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        scores = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % 2 == 0:  # Process most frames for smooth overlay
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image)

                if results.pose_landmarks:
                    # Draw skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

                    # Basic scoring for Side Kick
                    lm = results.pose_landmarks.landmark
                    left_knee_angle = self.calculate_angle(
                        np.array([lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]),
                        np.array([lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]),
                        np.array([lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y])
                    )

                    frame_score = 85 if 160 < left_knee_angle < 185 else 65
                    scores.append(frame_score)

            out.write(frame)

        cap.release()
        out.release()

        avg_score = int(np.mean(scores)) if scores else 70

        return {
            "overall_score": avg_score,
            "feedback": "Strong side kick! Focus on higher chamber and faster retraction." if avg_score > 75 else "Good effort. Work on chamber height and supporting leg stability.",
            "overlay_video_url": None,  # Will be filled by Railway URL later
            "technique": "Side Kick (Yeop Chagi)"
        }


analyzer = TaekwondoAnalyzer()


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    technique_type: str = Form("side_kick")
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, detail="Please upload a video file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
        temp_in.write(await file.read())
        temp_path = temp_in.name

    try:
        result = analyzer.process_video(temp_path, technique_type)
        
        # For now, return success. We'll handle overlay URL in Lovable later.
        return JSONResponse(content={
            "success": True,
            **result
        })
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health():
    return {"status": "healthy"}
    