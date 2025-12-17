import numpy as np

from surpRISe.components.ris import RIS


def sph2cart(spherical_coords: np.ndarray) -> np.ndarray:
    """
    Converts spherical coordinates to Cartesian coordinates.

    Parameters:
        spherical_coords: ndarray of shape (N, 3) with columns [R, phi_deg, theta_deg]

    Returns:
        cartesian_coords: ndarray of shape (N, 3) with columns [x, y, z]
    """
    spherical_coords = np.atleast_2d(spherical_coords)
    R = spherical_coords[:, 0]
    phi_deg = spherical_coords[:, 1]
    theta_deg = spherical_coords[:, 2]

    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)

    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    # Round near-zero values to 0.0
    tol = 1e-6
    x = np.where(np.abs(x) < tol, 0.0, x)
    y = np.where(np.abs(y) < tol, 0.0, y)
    z = np.where(np.abs(z) < tol, 0.0, z)

    return np.column_stack((x, y, z))


def cart2sph(cartesian_coords: np.ndarray) -> np.ndarray:
    """
    Converts Cartesian coordinates to spherical coordinates.

    Parameters:
        cartesian_coords: ndarray of shape (N, 3) with columns [x, y, z]

    Returns:
        spherical_coords: ndarray of shape (N, 3) with columns [R, phi_deg, theta_deg]
    """
    cartesian_coords = np.atleast_2d(cartesian_coords)
    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    z = cartesian_coords[:, 2]

    R = np.sqrt(x**2 + y**2 + z**2)

    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.where(R != 0, z / R, 0.0))
        phi = np.arctan2(y, x)

    phi_deg = np.degrees(phi)
    theta_deg = np.degrees(theta)

    # Round near-zero values to 0.0
    tol = 1e-6
    R = np.where(np.abs(R) < tol, 0.0, R)
    phi_deg = np.where(np.abs(phi_deg) < tol, 0.0, phi_deg)
    theta_deg = np.where(np.abs(theta_deg) < tol, 0.0, theta_deg)

    return np.column_stack((R, phi_deg, theta_deg))


import numpy as np


def dist_3d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the Euclidean distance between corresponding pairs of 3D Cartesian points.

    Parameters:
        A: ndarray of shape (N, 3), each row is a point [x, y, z]
        B: ndarray of shape (N, 3), each row is a point [x, y, z]

    Returns:
        distances: ndarray of shape (N,), Euclidean distances between A[i] and B[i]
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    if A.shape != B.shape or A.shape[1] != 3:
        raise ValueError("Both A and B must be of shape (N, 3)")

    return np.linalg.norm(A - B, axis=1)


def filt_round_decimals(
    array: np.ndarray, value: float, atol: float = 1e-12, rtol: float = 0
) -> np.ndarray:
    """
    Filters and rounds the elements of an array that are close to a specified value.

    Parameters:
        array: ndarray, input array to be filtered and rounded
        value: float, the value to which elements are compared
        atol: float, absolute tolerance for comparison
        rtol: float, relative tolerance for comparison

    Returns:
        array: the input array with elements close to the specified value rounded to their sign
    """

    mask = (np.abs(array) > value) & np.isclose(
        np.abs(array), value, atol=atol, rtol=rtol
    )
    array[mask] /= np.abs(array)[mask]

    return array


def ris_ff_border(
    ris: RIS = None,
    fc: float = None,
    phase_error_rad: float = np.pi / 8,
    phi_deg: float = None,
    theta_deg: float = None,
) -> float:
    """
    Computes the far-field border distance for a Reconfigurable Intelligent Surface (RIS).
    Parameters:
        ris : RIS
            Reconfigurable Intelligent Surface object (must be of class RIS).
        fc : float
            Carrier frequency in Hz.
        phase_error_rad : float
            Acceptable phase error in radians (default is pi/8).
        phi_deg : float
            Azimuth angle in degrees.
        theta_deg : float
            Elevation angle in degrees.

    Returns:
        r_border : float
            Far-field border distance in meters.
    """

    lam = 299792458 / fc
    d = np.maximum(ris.delta_h, ris.delta_v) / lam
    n = ris.Nrow
    m = ris.Ncol

    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)
    phase_error = np.abs(phase_error_rad)

    b = ((n - 1) / 2) ** 2 + ((m - 1) / 2) ** 2
    c = ((n - 1) / 2) * np.sin(theta) * np.cos(phi) + ((m - 1) / 2) * np.sin(
        theta
    ) * np.sin(phi)

    r_border = (
        (np.pi * b * d**2 - ((2 * np.pi * d * c - phase_error) ** 2 / (4 * np.pi)))
        * lam
        / phase_error
    )

    return r_border
