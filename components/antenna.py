import numpy as np

class Antenna:
    def __init__(self, position, gain_dbi):
        """
        Simple antenna class with position and gain attributes.

        Parameters:
            position (array-like): 3D coordinates of the antenna [x, y, z] in meters
            gain_dbi (float): Antenna gain in dBi
        """
        self.position = np.array(position, dtype=float)
        self.gain_dbi = gain_dbi

    def __repr__(self):
        return f"Antenna(position={self.position.tolist()}, gain_dbi={self.gain_dbi})"