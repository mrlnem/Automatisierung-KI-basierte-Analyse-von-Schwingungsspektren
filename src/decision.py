import numpy as np


def decide_nudge(probs):
    idx = int(np.argmax(probs))
    if idx == 0:
        return "uniform", "OK", "🟢"
    return ["severe", "rapid"][idx - 1], "DEFECT", "🔴"
