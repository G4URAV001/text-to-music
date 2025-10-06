import os
import time
import tempfile
from flask import Flask, request, jsonify, send_file, render_template

import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from scipy.io.wavfile import write as write_wav

app = Flask(__name__, template_folder="templates")

# Config - change via environment variables if desired
MODEL_NAME = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 7860))


def _load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {MODEL_NAME} onto device {device}...")
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.to(device)
    # set eval mode
    model.eval()
    print("Model loaded.")
    return model, processor, device


# load once at startup
MODEL, PROCESSOR, DEVICE = _load_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate audio from a text prompt.

    Request JSON: { "text": "A short prompt", "max_new_tokens": 256, "guidance_scale": 3, "do_sample": true }
    Returns: WAV file as attachment.
    """
    body = request.get_json(force=True)
    text = body.get('text', '')
    if not text or not text.strip():
        return jsonify({"error": "text prompt is required"}), 400

    # safe defaults + simple limits
    max_new_tokens = int(body.get('max_new_tokens', 256))
    guidance_scale = float(body.get('guidance_scale', 3.0))
    do_sample = bool(body.get('do_sample', True))

    # basic guardrails
    if max_new_tokens <= 0 or max_new_tokens > 3000:
        return jsonify({"error": "max_new_tokens must be between 1 and 3000"}), 400
    if guidance_scale < 0 or guidance_scale > 10:
        return jsonify({"error": "guidance_scale must be between 0 and 10"}), 400

    try:
        inputs = PROCESSOR(text=[text], padding=True, return_tensors="pt")
        # move inputs to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            audio_values = MODEL.generate(**inputs, do_sample=do_sample, guidance_scale=guidance_scale, max_new_tokens=max_new_tokens)

        # audio_values: (batch, channels, seq) or (batch, seq)
        audio_tensor = audio_values[0].cpu()

        # support both shapes
        if audio_tensor.ndim == 3:
            arr = audio_tensor[0].numpy()
        elif audio_tensor.ndim == 2:
            arr = audio_tensor.numpy()
        else:
            # fallback
            arr = audio_tensor.numpy()

        sampling_rate = MODEL.config.audio_encoder.sampling_rate

        # write to temporary .wav
        tmp_dir = tempfile.gettempdir()
        timestamp = int(time.time() * 1000)
        out_path = os.path.join(tmp_dir, f"musicgen_{timestamp}.wav")

        # ensure we write a 1D array (scipy expects shape (N,) or (N, channels))
        if arr.ndim == 1:
            write_wav(out_path, sampling_rate, arr)
        elif arr.ndim == 2:
            # (channels, seq) -> (seq, channels)
            write_wav(out_path, sampling_rate, arr.T)
        else:
            # try flatten
            write_wav(out_path, sampling_rate, arr.reshape(-1))

        return send_file(out_path, mimetype='audio/wav', as_attachment=True, download_name='generated.wav')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # debug? set FLASK_ENV=development
    app.run(host=HOST, port=PORT)
