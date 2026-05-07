"""
Segmentation Quality Reviewer (SSH / VSCode compatible)
Runs a local web server. VSCode will auto-forward the port and prompt
you to open it in your browser.

Usage:
    python review_masks.py <root_dir> [filename] [port]

    root_dir  : directory containing numbered subdirectories
    filename  : image filename to look for (default: masked.jpg)
    port      : port to serve on (default: 5000)

Controls (in browser):
    Right arrow / D  →  next
    Left  arrow / A  →  previous
    Space            →  play / pause
    Q                →  quit server
"""

import sys
import os
import glob
from flask import Flask, send_file, jsonify, request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
root  = sys.argv[1] if len(sys.argv) > 1 else "."
fname = sys.argv[2] if len(sys.argv) > 2 else "masked.jpg"
port  = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

pattern = os.path.join(root, "*", fname)
paths   = sorted(glob.glob(pattern))

if not paths:
    print(f"No images found matching: {pattern}")
    sys.exit(1)

print(f"Found {len(paths)} images.")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Segmentation Reviewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a1a1a;
    color: #e0e0e0;
    font-family: monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100vh;
    padding: 16px;
    gap: 12px;
  }
  #info { font-size: 13px; opacity: 0.6; letter-spacing: 0.04em; }
  #label { font-size: 18px; font-weight: bold; }
  #img-wrap {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    overflow: hidden;
  }
  img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 4px;
  }
  #controls { display: flex; gap: 16px; align-items: center; }
  button {
    background: #333;
    color: #eee;
    border: 1px solid #555;
    padding: 8px 24px;
    font-size: 14px;
    font-family: monospace;
    border-radius: 4px;
    cursor: pointer;
  }
  button:hover { background: #444; }
  #btn-play {
    background: #1a3a1a;
    border-color: #4a8a4a;
    color: #90e090;
    min-width: 90px;
  }
  #btn-play:hover { background: #224422; }
  #btn-play.playing {
    background: #3a1a1a;
    border-color: #8a4a4a;
    color: #e09090;
  }
  #btn-play.playing:hover { background: #442222; }
  #speed-wrap { display: flex; align-items: center; gap: 8px; font-size: 13px; opacity: 0.75; }
  #speed { width: 60px; background: #2a2a2a; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 4px 6px; font-family: monospace; font-size: 13px; text-align: center; }
</style>
</head>
<body>
  <div id="info">← / A &nbsp;|&nbsp; → / D &nbsp; to navigate &nbsp;|&nbsp; Space to play/pause &nbsp;|&nbsp; Q to quit server</div>
  <div id="label">Loading...</div>
  <div id="img-wrap"><img id="image" src="" alt="mask"></div>
  <div id="controls">
    <button onclick="move(-1)">&#9664; Prev</button>
    <button id="btn-play" onclick="togglePlay()">&#9654; Play</button>
    <button onclick="move(1)">Next &#9654;</button>
    <div id="speed-wrap">
      <label for="speed">fps</label>
      <input id="speed" type="number" value="2" min="0.1" max="30" step="0.5" onchange="resetInterval()">
    </div>
  </div>

<script>
  let idx = 0, total = 0;
  let playTimer = null;

  async function load() {
    const r = await fetch('/state');
    const d = await r.json();
    idx   = d.index;
    total = d.total;
    document.getElementById('label').textContent =
      '[' + (idx + 1) + ' / ' + total + ']  ' + d.folder;
    document.getElementById('image').src = '/image?t=' + Date.now();
  }

  async function move(delta) {
    await fetch('/move?delta=' + delta);
    load();
  }

  function fps() {
    return parseFloat(document.getElementById('speed').value) || 2;
  }

  function startPlay() {
    const btn = document.getElementById('btn-play');
    btn.textContent = '\u23F8 Pause';
    btn.classList.add('playing');
    playTimer = setInterval(() => move(1), 1000 / fps());
  }

  function stopPlay() {
    const btn = document.getElementById('btn-play');
    btn.textContent = '\u25B6 Play';
    btn.classList.remove('playing');
    clearInterval(playTimer);
    playTimer = null;
  }

  function togglePlay() {
    playTimer ? stopPlay() : startPlay();
  }

  function resetInterval() {
    if (playTimer) { stopPlay(); startPlay(); }
  }

  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === 'd') move(1);
    if (e.key === 'ArrowLeft'  || e.key === 'a') move(-1);
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    if (e.key === 'q') fetch('/quit');
  });

  load();
</script>
</body>
</html>
"""

state = {"index": 0}


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/state")
def get_state():
    i = state["index"]
    folder = os.path.basename(os.path.dirname(paths[i]))
    return jsonify(index=i, total=len(paths), folder=folder)


@app.route("/image")
def get_image():
    return send_file(paths[state["index"]])


@app.route("/move")
def move():
    delta = int(request.args.get("delta", 1))
    state["index"] = (state["index"] + delta) % len(paths)
    return ("", 204)


@app.route("/quit")
def quit_server():
    os.kill(os.getpid(), 9)


if __name__ == "__main__":
    print(f"Open http://localhost:{port} in your browser.")
    print("VSCode should auto-forward the port for you.")
    app.run(host="0.0.0.0", port=port, debug=False)