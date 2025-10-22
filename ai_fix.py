#!/usr/bin/env python3
# Auto Clip AI PRO — Versione v2 (ottimizzata: batch, mixed precision, prompt corretto)
# Autore: GPT-5

import os, cv2, threading, subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from PIL import Image

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

DEVICE = torch.device("cuda") if TORCH_AVAILABLE and torch.cuda.is_available() else torch.device("cpu") if TORCH_AVAILABLE else None
PROMPT_FILE = Path(__file__).with_name("last_prompt.txt")

# ========== Utility ==========

def ask_multiline_prompt(parent, title="Imposta Prompt AI", initial_text=""):
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.geometry("700x420")
    dlg.transient(parent)
    dlg.grab_set()

    tk.Label(dlg, text="Inserisci il prompt per l'AI:", font=("Arial", 11, "bold")).pack(pady=(6,0))
    text = tk.Text(dlg, wrap="word", font=("Consolas", 11))
    text.insert("1.0", initial_text)
    text.pack(fill="both", expand=True, padx=8, pady=8)

    result = {"value": None}

    btn_frame = tk.Frame(dlg)
    btn_frame.pack(fill="x", pady=6)

    def salva():
        result["value"] = text.get("1.0", "end-1c")
        dlg.destroy()

    def annulla():
        dlg.destroy()

    tk.Button(btn_frame, text="💾 Salva e Chiudi", command=salva).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Annulla", command=annulla).pack(side="left", padx=6)

    text.focus_set()
    dlg.wait_window()
    return result["value"]


def preprocess_frame_for_blip(frame, max_side=256):
    import numpy as np
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(img)


def init_blip_model(logger=None):
    if not TORCH_AVAILABLE:
        if logger: logger.write("[⚠] torch/transformers non disponibili.")
        return None, None
    try:
        if logger: logger.write("[🧠] Caricamento modello BLIP...")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        model.to(DEVICE).eval()
        if logger: logger.write(f"[✅] BLIP caricato su {DEVICE}.")
        return model, processor
    except Exception as e:
        if logger: logger.write(f"[❌] Errore caricando BLIP: {e}")
        return None, None


def analyze_video(video_path, model, processor, prompt_text, logger, progress, batch_size=4):
    if not os.path.exists(video_path): return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.write(f"[❌] Impossibile aprire video: {video_path}", "red")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    step = fps * 2  # 2 secondi tra i frame

    chosen, batch, indices = [], [], []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if idx % step == 0:
            img = preprocess_frame_for_blip(frame)
            batch.append(img)
            indices.append(idx)

            if len(batch) == batch_size:
                inputs = processor(images=batch, text=prompt_text or "describe scene", return_tensors="pt", padding=True).to(DEVICE)
                with torch.cuda.amp.autocast() if DEVICE.type=='cuda' else torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=30)
                captions = [processor.decode(o, skip_special_tokens=True) for o in out]

                for i, caption in zip(indices, captions):
                    if any(x in caption.lower() for x in ["person","action","scene","interesting","object"]):
                        logger.write(f"[✅ Scelto] {caption}", "green")
                        chosen.append((i, caption))
                    else:
                        logger.write(f"[⚪ Scartato] {caption}", "gray")
                progress["value"] = min(100, int(idx / total * 100))
                progress.update()
                batch, indices = [], []

        idx += 1

    # Process remaining batch
    if batch:
        inputs = processor(images=batch, text=prompt_text or "describe scene", return_tensors="pt", padding=True).to(DEVICE)
        with torch.cuda.amp.autocast() if DEVICE.type=='cuda' else torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        captions = [processor.decode(o, skip_special_tokens=True) for o in out]
        for i, caption in zip(indices, captions):
            if any(x in caption.lower() for x in ["person","action","scene","interesting","object"]):
                logger.write(f"[✅ Scelto] {caption}", "green")
                chosen.append((i, caption))
            else:
                logger.write(f"[⚪ Scartato] {caption}", "gray")

    cap.release()
    logger.write(f"[📊] Analisi completata. Scene scelte: {len(chosen)}")
    return chosen

# ========== GUI ==========

class GuiLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, msg, color=None):
        if not msg: return
        def append():
            tag = color
            if color and tag not in self.text_widget.tag_names():
                self.text_widget.tag_config(tag, foreground=color)
            self.text_widget.insert(tk.END, msg + "\n", tag)
            self.text_widget.see(tk.END)
        self.text_widget.after(0, append)

    def flush(self): pass

class AutoClipAI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Auto Clip AI PRO")
        self.geometry("1100x800")
        self.videos = []
        self.custom_prompt = self._load_prompt()
        self.model = self.processor = None
        self._create_widgets()
        self._load_model_thread()
        self._update_title()

    def _update_title(self):
        short = (self.custom_prompt[:50] + "…") if len(self.custom_prompt) > 50 else self.custom_prompt
        self.title(f"Auto Clip AI PRO — Prompt: {short or 'nessuno'}")

    def _save_prompt(self):
        try: PROMPT_FILE.write_text(self.custom_prompt, encoding="utf-8")
        except Exception: pass

    def _load_prompt(self):
        try: return PROMPT_FILE.read_text(encoding="utf-8").strip()
        except Exception: return ""

    def _create_widgets(self):
        top = tk.Frame(self); top.pack(fill="x", pady=8)
        tk.Button(top, text="➕ Aggiungi Video", command=self.add_video).pack(side="left", padx=4)
        tk.Button(top, text="🔍 Analizza Tutti", command=self.analyze_all).pack(side="left", padx=4)
        tk.Button(top, text="⏹ Ferma", command=self.stop_analysis).pack(side="left", padx=4)
        tk.Button(top, text="💾 Esporta Finale", command=self.export_video).pack(side="left", padx=4)
        tk.Button(top, text="🧠 Imposta Prompt AI", command=self.set_prompt).pack(side="left", padx=4)

        self.progress = ttk.Progressbar(self, orient="horizontal", length=1000, mode="determinate")
        self.progress.pack(pady=6)

        self.terminal = scrolledtext.ScrolledText(self, font=("Consolas", 11), height=25)
        self.terminal.pack(fill="both", expand=True, padx=8, pady=8)
        self.logger = GuiLogger(self.terminal)

    def _load_model_thread(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        self.model, self.processor = init_blip_model(self.logger)

    def add_video(self):
        files = filedialog.askopenfilenames(
            title="Seleziona video",
            filetypes=[
                ("Video MP4", "*.mp4"),
                ("Video MOV", "*.mov"),
                ("Video AVI", "*.avi"),
                ("Video MKV", "*.mkv"),
                ("Tutti i file", "*.*")
            ]
        )
        if not files: return
        self.videos.extend(files)
        for f in files:
            self.logger.write(f"[🎞] Aggiunto: {f}")

    def stop_analysis(self):
        self._stop = True
        self.logger.write("[⏹] Analisi interrotta.", "red")

    def analyze_all(self):
        if not self.model or not self.processor:
            self.logger.write("[⚠] Modello non pronto.")
            return
        if not self.videos:
            self.logger.write("[⚠] Nessun video selezionato.")
            return
        self._stop = False
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        self.progress["value"] = 0
        for v in self.videos:
            if getattr(self, '_stop', False): break
            self.logger.write(f"[▶] Analizzando {os.path.basename(v)}...", "cyan")
            analyze_video(v, self.model, self.processor, self.custom_prompt, self.logger, self.progress)
        self.logger.write("[✅] Analisi completate.")

    def set_prompt(self):
        new_p = ask_multiline_prompt(self, "Imposta Prompt AI", self.custom_prompt)
        if new_p is not None:
            self.custom_prompt = new_p.strip()
            self._save_prompt()
            self._update_title()
            self.logger.write(f"[🧠] Nuovo prompt impostato:\n{self.custom_prompt}", "cyan")

    def export_video(self):
        out = filedialog.asksaveasfilename(title="Esporta video finale", defaultextension=".mp4",
                                           filetypes=[("MP4","*.mp4")])
        if not out: return
        input_videos = " ".join(f'-i "{v}"' for v in self.videos)
        cmd = f'ffmpeg {input_videos} -filter_complex "concat=n={len(self.videos)}:v=1:a=1 [v][a]" -map "[v]" -map "[a]" "{out}" -y'
        self.logger.write(f"[💾] Eseguo FFmpeg:\n{cmd}", "gray")
        try:
            subprocess.run(cmd, shell=True, check=True)
            self.logger.write(f"[✅] Esportato: {out}", "green")
        except Exception as e:
            self.logger.write(f"[❌] Errore FFmpeg: {e}", "red")


def main():
    AutoClipAI().mainloop()

if __name__ == "__main__":
    main()

