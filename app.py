"""
Vector Swarm Flask API
Clean separation of Flask routes from business logic.
Run with: python -m flask run --debug
"""

import atexit
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from swarm_manager import SwarmManager, log
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for JS frontend

# Initialize the swarm manager
swarm = SwarmManager()


@app.before_request
def before_first_request():
    """Initialize swarm system before first request"""
    # This will only run once, even with multiple requests
    if not swarm.physics_engine:
        log("🚀 Initializing Vector Swarm System...", "FLASK")
        swarm.initialize()
        log("✓ System ready for operations", "FLASK")


# --- API Endpoints (Define first for priority) ---


@app.route("/api/swarm/start", methods=["POST"])
def start_swarm():
    """
    Start the swarm with a mission.
    Payload: { "mission": "Fix the DB deadlock", "agent_count": 3 }
    """
    data = request.json
    mission = data.get("mission", "Idling")
    count = int(data.get("agent_count", 3))

    result = swarm.start_swarm(mission, count)
    return jsonify(result)


@app.route("/api/swarm/stop", methods=["POST"])
def stop_swarm():
    """Stop all running agents"""
    swarm.stop_swarm()
    return jsonify({"status": "stopped"})


@app.route("/api/swarm/state", methods=["GET"])
def get_state():
    """
    Get current swarm state for visualization.
    Returns 3D coordinates for all agents + queen.
    """
    state = swarm.get_state()
    return jsonify(state)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "agents_running": len(swarm.agent_processes),
            "pca_fitted": swarm.pca_fitted,
        }
    )


# --- Frontend Routes (Catch-all last) ---


@app.route("/")
def index():
    """Serve the main UI"""
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static files (CSS, JS, images, etc.) - must be last route"""
    # Don't serve API routes through this
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(".", path)


# --- Cleanup Handler ---
def cleanup():
    """Cleanup on server shutdown"""
    log("Flask server shutdown - cleaning up resources", "FLASK")
    swarm.cleanup()


atexit.register(cleanup)


# --- Development Server (optional, prefer flask run) ---
if __name__ == "__main__":
    # This runs if you execute: python app.py
    # But it's better to use: python -m flask run --debug
    log("⚠️  Running Flask development server directly", "FLASK")
    log("💡 For better debugging, use: python -m flask run --debug", "FLASK")
    swarm.initialize()
    app.run(host="0.0.0.0", port=5000, debug=False)
