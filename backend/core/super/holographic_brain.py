"""
Holographic Brain - מוח הולוגרפי תלת-מימדי
"""
import math, random
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Neuron:
    id: int
    x: float
    y: float
    z: float
    activation: float
    connections: List[int]
    type: str

class HolographicBrain:
    def __init__(self):
        self.neurons = []
        self.layers = {
            "input": {"count": 8, "y": 100, "color": "#00d4ff"},
            "memory": {"count": 12, "y": 50, "color": "#a855f7"},
            "processing": {"count": 16, "y": 0, "color": "#00ffaa"},
            "output": {"count": 8, "y": -50, "color": "#ff7a00"},
        }
        self._build_brain()
        print(f"[HolographicBrain] 🧠 {len(self.neurons)} נוירונים")

    def _build_brain(self):
        nid = 0
        for lname, linfo in self.layers.items():
            count = linfo["count"]
            y = linfo["y"]
            for i in range(count):
                angle = (2*3.14159/count)*i
                radius = 80 + random.uniform(-10,10)
                x = math.cos(angle)*radius
                z = math.sin(angle)*radius
                conns = [random.randint(0,3) for _ in range(random.randint(2,4))]
                self.neurons.append(Neuron(nid, x, y, z, random.random(), conns, lname))
                nid+=1

    def activate(self, input_text: str) -> Dict:
        words = input_text.split()
        for i, n in enumerate([n for n in self.neurons if n.type=="input"]):
            n.activation = 0.8+random.random()*0.2 if i < len(words) else random.random()*0.3
        for lname in ["memory","processing","output"]:
            for n in [x for x in self.neurons if x.type==lname]:
                n.activation = min(1.0, random.random()*0.8+0.2)
        return {
            "neurons": [{"id": n.id, "x": round(n.x,1), "y": n.y, "z": round(n.z,1), "activation": round(n.activation,3), "type": n.type} for n in self.neurons],
            "active_neurons": len([n for n in self.neurons if n.activation>0.5]),
            "total_activation": round(sum(n.activation for n in self.neurons)/len(self.neurons),3)
        }

_global_holo = None
def get_holographic_brain():
    global _global_holo
    if _global_holo is None:
        _global_holo = HolographicBrain()
    return _global_holo
