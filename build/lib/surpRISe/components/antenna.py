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
        self.gain_linear = 10 ** (gain_dbi / 10)
        self.point_to(np.array([0.0, 0.0, 0.0]))  # Default pointing direction to origin

    def point_to(self, position_xyz):
        """
        Set the antenna pointing direction toward a given target position.

        Parameters:
            position_xyz (array-like): 3D target position [x, y, z] in meters
        """
        vec = np.array(position_xyz, dtype=float) - self.position
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Cannot point antenna to its own position.")
        self.pointing_direction = vec / norm  # Unit vector

    def get_rad_pat(self, positions_xyz):
        """
        Compute the antenna radiation pattern toward a set of 3D positions.

        Parameters:
            positions_xyz (ndarray): Nx3 array of 3D positions to evaluate the pattern.

        Returns:
            rad_pat (ndarray): Array of shape (N,) with gain (linear scale) at each direction.
        """
        vectors = positions_xyz - self.position  # shape: (N, 3)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / norms  # Normalize

        cos_ang = np.sum(unit_vectors * self.pointing_direction, axis=1)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        ang_rel = np.arccos(cos_ang)

        rad_pat = np.zeros_like(ang_rel)
        valid = (ang_rel >= 0) & (ang_rel <= np.pi / 2)
        rad_pat[valid] = np.minimum(1, np.cos(ang_rel[valid]) ** (self.gain_linear / 2 - 1)) # Limited to 1

        return rad_pat

    def __repr__(self):
        return f"Antenna(position={self.position.tolist()}, gain_dbi={self.gain_dbi}, pointing_direction={self.pointing_direction.tolist()})"