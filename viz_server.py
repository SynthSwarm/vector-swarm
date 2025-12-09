import numpy as np
from flask import Flask, jsonify
from sklearn.decomposition import PCA
from swarm_physics import SwarmPhysics

app = Flask(__name__)

class SwarmVisualizer:
    def __init__(self, physics_engine: SwarmPhysics):
        self.physics = physics_engine
        # We keep a persistent PCA model to minimize "jitter" between frames
        # though with dynamic data, some re-fitting is inevitable.
        self.pca = PCA(n_components=3)

    def get_snapshot(self):
        """
        Reads raw high-dim vectors from shared memory and projects them to 3D.
        """
        # 1. Read Raw Memory
        # We need to know how many items are actually active
        # The counter at offset 0 tells us the limit
        count = self.physics.space.counter_view[0]
        
        if count < 3:
            # Not enough points for 3D PCA? Return dummy data or handle gracefully
            return {"status": "waiting_for_entropy", "data": []}

        # 2. Extract Vectors (Shape: N x 768)
        # Slicing the numpy array is zero-copy and fast
        raw_vectors = self.physics.space.vector_view[:count]
        
        # 3. Dimensionality Reduction (768 -> 3)
        # We re-fit on every frame to adapt to the shifting context space.
        # In production, you might partial_fit or freeze the model after initialization.
        reduced_vectors = self.pca.fit_transform(raw_vectors)
        
        # 4. Format for Frontend
        snapshot = []
        
        # Index 0 is ALWAYS the Queen (The Mission)
        queen_pos = reduced_vectors[0]
        snapshot.append({
            "id": "QUEEN",
            "type": "queen",
            "x": float(queen_pos[0]),
            "y": float(queen_pos[1]),
            "z": float(queen_pos[2]),
            # In a real app, fetch the Queen's actual text from metadata dict
            "label": "MISSION: Resolve Deadlock" 
        })
        
        # Indices 1..N are the Drones (Agents)
        for i in range(1, count):
            pos = reduced_vectors[i]
            snapshot.append({
                "id": f"Agent-{i-1}",
                "type": "drone",
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "label": f"Drone {i-1}" # You'd grab 'current_task' from metadata here
            })
            
        return {"status": "active", "data": snapshot}

# --- Initialization ---
# Connect to the EXISTING shared memory (Create=False)
# Ensure your main swarm script is running first!
physics_engine = SwarmPhysics(create=False)
visualizer = SwarmVisualizer(physics_engine)

@app.route('/api/swarm/state', methods=['GET'])
def get_swarm_state():
    try:
        data = visualizer.get_snapshot()
        # Add CORS headers if you are testing from a separate frontend
        response = jsonify(data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on port 5000
    print("--- 📡 Vector Swarm Visualizer Uplink Active 📡 ---")
    app.run(host='0.0.0.0', port=5000, debug=False)