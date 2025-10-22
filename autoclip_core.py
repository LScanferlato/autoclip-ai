import os, cv2, tempfile, subprocess, torch, numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_SKIP = 30
CLIP_LOCAL_DIR = "./clip_model_local"
CLIP_SIMILARITY_THRESHOLD = 0.95
SCENE_CHANGE_THRESHOLD = 0.35
BATCH_SIZE = 16
FFMPEG_CRF = 23
FFMPEG_PRESET = "fast"

class AutoClipCore:
    def __init__(self):
        self.blip_model = None
        self.blip_processor = None
        self.clip_model = None
        self.prompts = ["paesaggio cinematografico", "scena emozionante"]
        self.weight_blip = 0.5
        self.tempdir = tempfile.mkdtemp()

    def load_models(self):
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        self.clip_model = SentenceTransformer('clip-ViT-B-32', cache_folder=CLIP_LOCAL_DIR, device=str(DEVICE))

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % FRAME_SKIP == 0:
                frames.append(frame.copy())
            idx += 1
        cap.release()
        return frames

    def ffmpeg_has_nvenc(self):
        try:
            res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
            out = res.stdout.lower()
            return ("h264_nvenc" in out) or ("hevc_nvenc" in out)
        except Exception:
            return False

    def export_video(self, video_path, selected_indices, filename="video_finale"):
        fps = 30
        intervals = []
        for group in selected_indices:
            start = max(group[0] * FRAME_SKIP / fps, 0)
            end = (group[-1] + 1) * FRAME_SKIP / fps
            intervals.append((start, end))

        temp_clips = []
        codec = "h264_nvenc" if self.ffmpeg_has_nvenc() else "libx264"

        for i, (start, end) in enumerate(intervals, 1):
            temp_clip = os.path.join(self.tempdir, f"clip_{i:04d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-to", str(end),
                "-i", video_path,
                "-vf", "fps=30,scale=1280:-2",
                "-c:v", codec,
                "-preset", FFMPEG_PRESET,
                "-crf", str(FFMPEG_CRF),
                "-c:a", "aac", "-b:a", "128k",
                temp_clip
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            temp_clips.append(temp_clip)

        list_file = os.path.join(self.tempdir, "clips.txt")
        with open(list_file, "w") as f:
            for clip in temp_clips:
                f.write(f"file '{clip}'\n")

        final_out = os.path.join(os.getcwd(), f"{filename}.mp4")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", final_out])
        return final_out
