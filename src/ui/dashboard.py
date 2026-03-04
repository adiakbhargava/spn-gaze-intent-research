"""
Real-Time Demo Dashboard

Flask + WebSocket-based dashboard that visualizes the live intent decoding
pipeline. Shows:

1. **Target Grid:** 3x3 grid of selectable targets with gaze cursor overlay
2. **Classification Indicator:** Real-time intent vs. observe prediction
3. **Confidence Meter:** Classification probability
4. **Latency Panel:** Pipeline timing breakdown
5. **Accuracy Tracker:** Running classification accuracy

The dashboard receives updates via SocketIO from the inference engine
and renders them in-browser for easy screen recording (for the demo video).
"""

from __future__ import annotations

import json
import logging
import time
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from flask import Flask, render_template_string
    from flask_socketio import SocketIO
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask/flask-socketio not available")


# HTML template for the dashboard (self-contained, no external files needed)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intent Stream Pipeline — Live Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            background: #0a0a0f;
            color: #e0e0e0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        .main-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 10px 0 20px;
        }
        .header h1 {
            font-size: 1.4em;
            color: #00d4ff;
            font-weight: 400;
            letter-spacing: 2px;
        }
        .header .subtitle {
            font-size: 0.75em;
            color: #666;
            margin-top: 4px;
        }
        .grid-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        .target-grid {
            display: grid;
            grid-template-columns: repeat(3, 120px);
            grid-template-rows: repeat(3, 120px);
            gap: 12px;
            position: relative;
        }
        .target {
            background: #1a1a2e;
            border: 2px solid #2a2a4a;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            transition: all 0.15s ease;
            position: relative;
        }
        .target.active {
            border-color: #00d4ff;
            background: #0a2a3f;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        .target.selected {
            border-color: #00ff88;
            background: #0a3f2a;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.4);
            animation: pulse 0.4s ease;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .gaze-cursor {
            position: absolute;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: rgba(255, 100, 100, 0.8);
            border: 2px solid #ff4444;
            transform: translate(-50%, -50%);
            pointer-events: none;
            transition: left 0.05s linear, top 0.05s linear;
            z-index: 10;
        }
        .sidebar {
            width: 320px;
            background: #0f0f1a;
            border-left: 1px solid #1a1a2e;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow-y: auto;
        }
        .panel {
            background: #1a1a2e;
            border-radius: 8px;
            padding: 16px;
        }
        .panel-title {
            font-size: 0.7em;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #666;
            margin-bottom: 12px;
        }
        .prediction {
            text-align: center;
            padding: 20px;
        }
        .prediction .label {
            font-size: 1.8em;
            font-weight: 700;
        }
        .prediction .label.intent { color: #00ff88; }
        .prediction .label.observe { color: #ff6666; }
        .confidence-bar {
            height: 8px;
            background: #2a2a4a;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 12px;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6666, #ffaa00, #00ff88);
            border-radius: 4px;
            transition: width 0.15s ease;
        }
        .confidence-value {
            text-align: center;
            font-size: 2em;
            margin-top: 8px;
            font-weight: 300;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 0.85em;
        }
        .stat-label { color: #888; }
        .stat-value { color: #00d4ff; font-weight: 600; }
        .accuracy-value {
            font-size: 2.5em;
            text-align: center;
            font-weight: 300;
            color: #00d4ff;
        }
        .trial-counter {
            text-align: center;
            font-size: 0.85em;
            color: #666;
            padding-top: 8px;
        }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .status-dot.live { background: #00ff88; animation: blink 1s infinite; }
        .status-dot.offline { background: #ff4444; }
        @keyframes blink { 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="main-panel">
        <div class="header">
            <h1>INTENT STREAM PIPELINE</h1>
            <div class="subtitle">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">Connecting...</span>
                &nbsp;|&nbsp; Gaze-Guided Intent Decoding via EEG+Eye-Tracking Fusion
            </div>
        </div>
        <div class="grid-container">
            <div class="target-grid" id="targetGrid">
                <div class="target" data-idx="0">1</div>
                <div class="target" data-idx="1">2</div>
                <div class="target" data-idx="2">3</div>
                <div class="target" data-idx="3">4</div>
                <div class="target" data-idx="4">5</div>
                <div class="target" data-idx="5">6</div>
                <div class="target" data-idx="6">7</div>
                <div class="target" data-idx="7">8</div>
                <div class="target" data-idx="8">9</div>
            </div>
            <div class="gaze-cursor" id="gazeCursor"></div>
        </div>
    </div>
    <div class="sidebar">
        <div class="panel prediction">
            <div class="panel-title">Classification</div>
            <div class="label" id="predLabel">—</div>
        </div>
        <div class="panel">
            <div class="panel-title">Confidence</div>
            <div class="confidence-value" id="confValue">0.00</div>
            <div class="confidence-bar">
                <div class="confidence-fill" id="confFill" style="width: 0%"></div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-title">Running Accuracy</div>
            <div class="accuracy-value" id="accuracy">—</div>
            <div class="trial-counter" id="trialCounter">0 predictions</div>
        </div>
        <div class="panel">
            <div class="panel-title">Pipeline Latency</div>
            <div class="stat-row">
                <span class="stat-label">Total</span>
                <span class="stat-value" id="latTotal">— ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Features</span>
                <span class="stat-value" id="latFeat">— ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Model</span>
                <span class="stat-value" id="latModel">— ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Mean (total)</span>
                <span class="stat-value" id="latMean">— ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">P95 (total)</span>
                <span class="stat-value" id="latP95">— ms</span>
            </div>
        </div>
        <div class="panel">
            <div class="panel-title">Stream Info</div>
            <div class="stat-row">
                <span class="stat-label">Trial</span>
                <span class="stat-value" id="trialIdx">—</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Gaze Position</span>
                <span class="stat-value" id="gazePos">—</span>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
    <script>
        const socket = io();
        const grid = document.getElementById('targetGrid');
        const cursor = document.getElementById('gazeCursor');
        const targets = document.querySelectorAll('.target');
        let predCount = 0;
        let correctCount = 0;

        socket.on('connect', () => {
            document.getElementById('statusDot').className = 'status-dot live';
            document.getElementById('statusText').textContent = 'Live';
        });

        socket.on('disconnect', () => {
            document.getElementById('statusDot').className = 'status-dot offline';
            document.getElementById('statusText').textContent = 'Disconnected';
        });

        socket.on('prediction', (data) => {
            // Update prediction label
            const label = document.getElementById('predLabel');
            if (data.prediction === 1) {
                label.textContent = 'INTENT';
                label.className = 'label intent';
            } else {
                label.textContent = 'OBSERVE';
                label.className = 'label observe';
            }

            // Update confidence
            const conf = data.confidence;
            document.getElementById('confValue').textContent = conf.toFixed(3);
            document.getElementById('confFill').style.width = (conf * 100) + '%';

            // Update latency
            document.getElementById('latTotal').textContent = data.latency_ms.toFixed(1) + ' ms';
            document.getElementById('latFeat').textContent = data.feature_latency_ms.toFixed(1) + ' ms';
            document.getElementById('latModel').textContent = data.model_latency_ms.toFixed(1) + ' ms';

            // Update gaze cursor position
            updateGazeCursor(data.gaze_x, data.gaze_y);
            document.getElementById('gazePos').textContent =
                `(${data.gaze_x.toFixed(1)}, ${data.gaze_y.toFixed(1)})`;

            // Update target highlighting
            updateTargets(data.gaze_x, data.gaze_y, data.prediction === 1 && conf > 0.7);

            predCount++;
        });

        socket.on('latency_stats', (data) => {
            if (data.total) {
                document.getElementById('latMean').textContent = data.total.mean_ms.toFixed(1) + ' ms';
                document.getElementById('latP95').textContent = data.total.p95_ms.toFixed(1) + ' ms';
            }
        });

        socket.on('accuracy', (data) => {
            document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
            document.getElementById('trialCounter').textContent = data.n_predictions + ' predictions';
        });

        socket.on('trial_info', (data) => {
            document.getElementById('trialIdx').textContent =
                `${data.trial_idx + 1} / ${data.total_trials}`;
        });

        function updateGazeCursor(x, y) {
            const rect = grid.getBoundingClientRect();
            // Map gaze coordinates (-15 to 15) to grid pixel coordinates
            const px = rect.left + rect.width * (x + 15) / 30;
            const py = rect.top + rect.height * (y + 15) / 30;
            cursor.style.left = px + 'px';
            cursor.style.top = py + 'px';
        }

        function updateTargets(gx, gy, isClick) {
            // Determine which target the gaze is nearest
            const targetPositions = [
                [-8, -8], [0, -8], [8, -8],
                [-8, 0],  [0, 0],  [8, 0],
                [-8, 8],  [0, 8],  [8, 8],
            ];
            let minDist = Infinity;
            let nearest = -1;
            targetPositions.forEach((pos, idx) => {
                const d = Math.sqrt((gx - pos[0])**2 + (gy - pos[1])**2);
                if (d < minDist) { minDist = d; nearest = idx; }
            });

            targets.forEach((t, idx) => {
                t.classList.remove('active', 'selected');
                if (idx === nearest && minDist < 6) {
                    t.classList.add('active');
                    if (isClick) {
                        t.classList.add('selected');
                    }
                }
            });
        }
    </script>
</body>
</html>
"""


class DemoDashboard:
    """
    Flask+SocketIO dashboard for real-time intent visualization.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and flask-socketio required. pip install flask flask-socketio")

        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "intent-stream-pipeline"
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")
        self._thread = None

        @self.app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML)

        self._n_predictions = 0
        self._n_correct = 0

    def start(self):
        """Start the dashboard server in a background thread."""
        self._thread = threading.Thread(
            target=lambda: self.socketio.run(
                self.app, host=self.host, port=self.port,
                allow_unsafe_werkzeug=True, log_output=False,
            ),
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Dashboard running at http://{self.host}:{self.port}")

    def send_prediction(self, tick):
        """Send an inference tick to the dashboard."""
        self.socketio.emit("prediction", {
            "prediction": tick.prediction,
            "confidence": tick.confidence,
            "gaze_x": tick.gaze_x,
            "gaze_y": tick.gaze_y,
            "latency_ms": tick.latency_ms,
            "feature_latency_ms": tick.feature_latency_ms,
            "model_latency_ms": tick.model_latency_ms,
        })

    def send_latency_stats(self, stats: dict):
        """Send latency statistics to the dashboard."""
        self.socketio.emit("latency_stats", stats)

    def send_accuracy(self, accuracy: float, n_predictions: int):
        """Send running accuracy to the dashboard."""
        self.socketio.emit("accuracy", {
            "accuracy": accuracy,
            "n_predictions": n_predictions,
        })

    def send_trial_info(self, trial_idx: int, total_trials: int):
        """Send current trial information."""
        self.socketio.emit("trial_info", {
            "trial_idx": trial_idx,
            "total_trials": total_trials,
        })

    def stop(self):
        """Stop the dashboard."""
        # Flask-SocketIO doesn't have a clean shutdown from a thread,
        # but since it's a daemon thread it will stop with the main process
        pass
