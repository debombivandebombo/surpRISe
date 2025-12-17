import numpy as np


class Antenna:
    def __init__(self, position: list, gain_dbi: float) -> None:
        """
        Simple antenna class with position and gain attributes.

        Parameters:
            position (array-like): 3D coordinates of the antenna [x, y, z] in meters
            gain_dbi (float): Antenna gain in dBi
        """
        self.position: np.ndarray = np.array(position, dtype=float)
        self.gain_dbi: float = gain_dbi
        self.gain_linear: float = 10 ** (gain_dbi / 10)
        self.point_to(np.array([0.0, 0.0, 0.0]))  # Default pointing direction to origin
        self.defined_rad_pat: bool = False
        self.rad_pat: dict = {}

    def point_to(self, position_xyz: np.ndarray) -> None:
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

    def get_rad_pat(self, positions_xyz: np.ndarray) -> np.ndarray:
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

        if self.defined_rad_pat == False:
            valid = (ang_rel >= 0) & (ang_rel <= np.pi / 2)
            rad_pat[valid] = np.minimum(
                1, np.cos(ang_rel[valid]) ** (self.gain_linear / 2 - 1)
            )  # Limited to 1
        else:
            # Convert ang_rel from radians to degrees
            ang_rel_deg = np.degrees(ang_rel)
            angles = self.rad_pat["angles"]
            values = self.rad_pat["values"]

            # For each angle, find the closest defined angle and interpolate if needed
            for i, ang in enumerate(ang_rel_deg):
                idx = np.argmin(np.abs(angles - ang))
                diff = np.abs(angles[idx] - ang)
                if diff <= 2:
                    rad_pat[i] = values[idx]
                else:
                    # Interpolate using np.interp
                    rad_pat[i] = np.interp(ang, angles, values)

        return rad_pat

    def define_rad_pat(self, angles: list, values: list) -> None:
        """
        Define normalized radiation pattern for the antenna for the given azimuth and elevation angles.

        Parameters:
            angles (array-like): Vector of relative angles to the boresight (in degrees). Should be in crescent order.
            values (array-like): Vector of normalized linear values for each combination of angles.
        """
        angles = np.array(angles)
        values = np.array(values)
        if values.shape != (len(angles),):
            raise ValueError("Array 'values' must have dimension (len(angles),)")

        if np.min(angles) > 0 or np.max(angles) < 180:
            raise ValueError(
                "Array 'angles' must cover at least from 0 to 180 degrees."
            )
        if 0 not in angles or 180 not in angles:
            raise ValueError("Array 'angles' must include both 0 and 180 degrees.")

        max_val = np.max(values)
        if max_val > 1:
            values = values / max_val

        # Sort angles and values in increasing order of angles
        sort_idx = np.argsort(angles)
        angles = angles[sort_idx]
        values = values[sort_idx]

        self.rad_pat = {"angles": angles, "values": values}

        self.defined_rad_pat = True

    def __repr__(self) -> str:
        return f"Antenna(position={self.position.tolist()}, gain_dbi={self.gain_dbi}, pointing_direction={self.pointing_direction.tolist()})"
