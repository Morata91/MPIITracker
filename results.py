from dataclasses import dataclass
import numpy as np

@dataclass
class GazeResultContainer:

    x: np.ndarray
    y: np.ndarray
