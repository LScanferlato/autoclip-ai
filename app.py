from flask import Flask, request, jsonify, render_template
from autoclip_core import AutoClipCore
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

core = AutoClipCore()
core.load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files.get('video')
    if not file or file.filename == '':
        return jsonify({'error': 'No video uploaded'}), 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    return jsonify({'message': 'Video uploaded', 'path': path})

@app.route('/analyze', methods=['POST'])
def analyze():
    path = request.json.get('path')
    frames = core.extract_frames(path)
    # Qui potresti aggiungere analisi con BLIP/CLIP
    selected = [[0, 1, 2], [10, 11, 12]]  # Esempio: gruppi di frame selezionati
    output_path = core.export_video(path, selected)
    return jsonify({'status': 'Analisi completata', 'output': output_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
