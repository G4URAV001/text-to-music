MusicGen simple web frontend
=================================

This small project wraps the MusicGen model (from the notebook) in a minimal Flask web UI. It allows you to provide a text prompt and download a generated WAV file.

Files added:
- `web_app.py` — Flask app that loads `facebook/musicgen-small` by default and exposes `/generate` endpoint.
- `templates/index.html` — Simple single-page UI to enter prompts and play/download results.
- `requirements.txt` — Python dependencies.

Quick start (recommended: use WSL/Ubuntu environment as in your workspace):

1) Create and activate a Python virtual environment (replace python with your python3 path if needed):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2) Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Note: Installing `torch` via `pip install torch` may not pick the best CUDA build for your GPU. If you have a CUDA-capable GPU, follow official instructions at https://pytorch.org to install the correct package (for example: `pip install torch --index-url https://download.pytorch.org/whl/cu121` or similar depending on CUDA version).

3) Run the Flask app:

```powershell
python web_app.py
```

4) Open your browser to http://127.0.0.1:7860/ and use the UI.

Notes and troubleshooting:
- The first model load downloads the pretrained weights and may take several minutes depending on your connection and cache.
- If you run on CPU, generation may be slow. For faster generation use a GPU (CUDA) and install a compatible torch build.
- If you want to use a different checkpoint (e.g., `facebook/musicgen-medium`), set the environment variable `MUSICGEN_MODEL` before launching the app.

Example to run with a different model and custom port:

```powershell
$env:MUSICGEN_MODEL = 'facebook/musicgen-medium'; $env:PORT = '8000'; python web_app.py
```

Security: This app is a local demo. Do not expose it publicly without adding authentication and proper resource controls.

# change to the project directory
cd /home/aura/ai

# create a virtual environment (use python3 if you have both)
python3 -m venv venv

# activate it
source venv/bin/activate

# upgrade pip
pip install --upgrade pip

# install dependencies from requirements.txt
pip install -r requirements.txt

# set Flask env (optional: enables debug mode)
export FLASK_APP=web_app.py
export FLASK_ENV=development

# run the Flask app so it's reachable from Windows as http://localhost:5000
flask run --host=0.0.0.0 --port=5000
