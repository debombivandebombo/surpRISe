import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi

from surpRISe.utils.utils import *
from surpRISe.components.antenna import Antenna
from surpRISe.components.ris import RIS

def calc_spherical_path_exp_shift(tx_pos: np.ndarray, rx_pos: np.ndarray, ris_positions: np.ndarray, fc: float) -> np.ndarray:
    """
    Compute the total exp phase shift from Tx to RIS to Rx
    under the spherical wavefront assumption.

    Parameters:
    -----------
    tx_pos : array-like of shape (3,)
        Transmitter position in Cartesian coordinates (x, y, z).
    rx_pos : array-like of shape (3,)
        Receiver position in Cartesian coordinates (x, y, z).
    ris_positions : array of shape (Nrow, Ncol, 3)
        Positions of each RIS element in Cartesian coordinates.
    fc : float
        Carrier frequency in Hz.

    Returns:
    --------
    total_exp_shift : complex array of shape (Nrow, Ncol)
        Complex exponential phase term for each RIS element.
    """
    wavelength = c / fc

    tx_pos = np.array(tx_pos).reshape(1, 1, 3)
    rx_pos = np.array(rx_pos).reshape(1, 1, 3)

    dist_tx2ris = np.linalg.norm(ris_positions - tx_pos, axis=2)
    dist_ris2rx = np.linalg.norm(ris_positions - rx_pos, axis=2)

    total_distance = dist_tx2ris + dist_ris2rx
    total_exp_shift = np.exp(-1j * 2 * pi * total_distance / wavelength)

    return total_exp_shift

def calc_plane_path_exp_shift(tx_pos: np.ndarray, rx_pos: np.ndarray, ris: RIS, fc: float) -> np.ndarray:
    """
    Compute the total exp phase shift from Tx to RIS to Rx
    under the plane wavefront assumption.

    The approximation considers that the incident and reflected
    waves can be modeled by their directions (unit vectors)
    relative to the RIS center, instead of individual distances.

    Parameters:
    -----------
    tx_pos : array-like of shape (3,)
        Transmitter position in Cartesian coordinates (x, y, z).
    rx_pos : array-like of shape (3,)
        Receiver position in Cartesian coordinates (x, y, z).
    ris : object of type RIS
        RIS object.
    fc : float
        Carrier frequency in Hz.

    Returns:
    --------
    total_exp_shift : complex array of shape (Nrow, Ncol)
        Complex exponential phase term for each RIS element.
    """
    wavelength = c / fc
    k = 2 * pi / wavelength

    # Calculate RIS center and positions
    ris_center = ris.center.copy()  # shape: (3,)
    ris_positions = ris.positions.copy()  # shape: (Nrow, Ncol, 3)

    tx_pos = tx_pos.reshape(3,)
    rx_pos = rx_pos.reshape(3,)

    # Unitary direction vectors from Tx to RIS and RIS to Rx
    u_tx = (ris_center - tx_pos)
    u_tx /= np.linalg.norm(u_tx)

    u_rx = (rx_pos - ris_center)
    u_rx /= np.linalg.norm(u_rx)

    # Effective wave vector
    k_eff = u_tx - u_rx

    # Relative positions of RIS elements to RIS center
    ris_rel_pos = ris_positions - ris_center

    # Scalar product to get phase shifts
    phase_shifts = k * np.tensordot(ris_rel_pos, k_eff, axes=([2], [0]))

    total_exp_shift = np.exp(-1j * phase_shifts)

    return total_exp_shift

def calc_total_path_exp_shift(tx_antenna: Antenna = None, rx_antenna: Antenna = None, ris: RIS = None, wave_model: str = 'spherical') -> np.ndarray:
    """
    Computethe total exp phase shift for each RIS element
    depending on the wave model.

    Parameters:
    -----------
    tx_pos : array-like of shape (3,)
        Transmitter position in Cartesian coordinates (x, y, z).
    rx_pos : array-like of shape (3,)
        Receiver position in Cartesian coordinates (x, y, z).
    ris : object of type RIS
        RIS object.
    wave_model : str
        Wave model to use ('spherical' or 'plane').

    Returns:
    --------
    total_path_exp_shift : complex array of shape (Nrow, Ncol)
        Total complex exponential phase term for each RIS element.
    """
    # Compute total path exponential shift depending on wave model
    if wave_model == "spherical":
        total_path_exp_shift = calc_spherical_path_exp_shift(
            tx_pos=tx_antenna.position,
            rx_pos=rx_antenna.position,
            ris_positions=ris.positions,
            fc=ris.fc
        )

    elif wave_model == "plane":
        total_path_exp_shift = calc_plane_path_exp_shift(
            tx_antenna.position,
            rx_antenna.position,
            ris,
            ris.fc
        )

    else:
        raise ValueError("Invalid wave_model. Use 'spherical' or 'plane'.")
    
    return total_path_exp_shift

def calc_tx_ris_rx_cont_opt(tx_antenna: Antenna = None, rx_antenna: Antenna = None, ris: RIS = None, wave_model: str = 'spherical') -> np.ndarray:
    """
    Compute optimal continuous phase shift for each RIS element.

    Parameters:
    -----------
    tx_pos : array-like of shape (3,)
        Transmitter position in Cartesian coordinates (x, y, z).
    rx_pos : array-like of shape (3,)
        Receiver position in Cartesian coordinates (x, y, z).
    ris : object of type RIS
        RIS object.
    wave_model : str
        Wave model to use ('spherical' or 'plane').

    Returns:
    --------
    desired_complex : complex array of shape (Nrow, Ncol)
        Desired complex term for each RIS element.
    desired_phase : complex array of shape (Nrow, Ncol)
        Desired phase term for each RIS element.
    """
    # Compute total path exponential shift depending on wave model
    total_path_exp_shift = calc_total_path_exp_shift(tx_antenna=tx_antenna, rx_antenna=rx_antenna, ris=ris, wave_model=wave_model)

    # Calculate continuous (perfect) phase shift for each RIS element
    desired_complex = total_path_exp_shift ** -1
    desired_phase = np.angle(desired_complex)

    return desired_complex, desired_phase

class tx_ris_rx_direct_disc_opt:
    def __init__(self, tx_antenna: Antenna = None, rx_antenna: Antenna = None, ris: RIS = None, wave_model: str = 'spherical') -> None:
        """
        Parameters:
        -----------
        tx_antenna : Antenna object
            Transmitter antenna object (must be provided).
        rx_antenna : Antenna object
            Receiver antenna object (must be provided).
        ris : RIS object
            Reconfigurable Intelligent Surface object (must be provided).
        wave_model : str
            Wave model to use ('spherical' or 'plane').
        """
        if tx_antenna is None:
            raise ValueError("tx_antenna must be provided")
        if rx_antenna is None:
            raise ValueError("rx_antenna must be provided")
        if ris is None:
            raise ValueError("ris must be provided")

        self.tx_antenna = tx_antenna
        self.rx_antenna = rx_antenna
        self.ris = ris
        self.wave_model = wave_model.lower()

    def optimize(self) -> np.ndarray:
        """
        Compute optimal state (phase shift) for each RIS element.

        - mode:
            "cont" => exact inverse phase (continuous phase adjustment).
            "disc" => quantized phase to nearest available RIS state.
        - wave_model:
            "spherical" => phases computed assuming spherical wavefronts.
            "plane"     => phases computed assuming plane wavefronts.
        """
        
        desired_complex, desired_phase = calc_tx_ris_rx_cont_opt(
            tx_antenna=self.tx_antenna, 
            rx_antenna=self.rx_antenna, 
            ris=self.ris, 
            wave_model=self.wave_model
        )

        # Discretize phase to nearest value in phase_shifts_dict
        discrete_phase = np.zeros_like(desired_phase, dtype=float)
        # Valid shifts from dictionary
        valid_discrete_shifts = np.array(list(self.ris.phase_shifts_dict.values()))
        for i in range(self.ris.Nrow):
            for j in range(self.ris.Ncol):
                phase_diffs = np.abs(np.angle(np.exp(1j * valid_discrete_shifts) / desired_complex[i, j]))
                idx = np.argmin(phase_diffs)
                state_key = list(self.ris.phase_shifts_dict.keys())[idx]
                discrete_phase[i, j] = self.ris.phase_shifts_dict[state_key]
        return discrete_phase
            
class tx_ris_rx_max_bf_opt:
    def __init__(self, tx_antenna: Antenna = None, rx_antenna: Antenna = None, ris: RIS = None, wave_model: str = 'spherical') -> None:
        """
        Parameters:
        -----------
        tx_antenna : Antenna object
            Transmitter antenna object (must be provided).
        rx_antenna : Antenna object
            Receiver antenna object (must be provided).
        ris : RIS object
            Reconfigurable Intelligent Surface object (must be provided).
        wave_model : str
            Wave model to use ('spherical' or 'plane').
        """
        if tx_antenna is None:
            raise ValueError("tx_antenna must be provided")
        if rx_antenna is None:
            raise ValueError("rx_antenna must be provided")
        if ris is None:
            raise ValueError("ris must be provided")

        self.tx_antenna = tx_antenna
        self.rx_antenna = rx_antenna
        self.ris = ris
        self.wave_model = wave_model.lower()

    def optimize(self) -> np.ndarray:
        """
        Compute optimal state (phase shift) for each RIS element.

        - mode:
            "cont" => exact inverse phase (continuous phase adjustment).
            "disc" => quantized phase to nearest available RIS state.
        - wave_model:
            "spherical" => phases computed assuming spherical wavefronts.
            "plane"     => phases computed assuming plane wavefronts.
        """

        desired_complex, desired_phase = calc_tx_ris_rx_cont_opt(
            tx_antenna=self.tx_antenna, 
            rx_antenna=self.rx_antenna, 
            ris=self.ris, 
            wave_model=self.wave_model
        )

        total_path_exp_shift = desired_complex ** -1

        valid_discrete_shifts = np.array(list(self.ris.phase_shifts_dict.values()))

        # Grid search
        alphas = np.linspace(0, 2 * np.pi, 360)
        best_bf = -np.inf
        best_alpha = 0

        for alpha in alphas:
            desired_complex_alpha = desired_complex * np.exp(1j * alpha)
            discrete_phase = np.zeros_like(desired_phase, dtype=float)

            for i in range(self.ris.Nrow):
                for j in range(self.ris.Ncol):
                    phase_diffs = np.abs(np.angle(np.exp(1j * valid_discrete_shifts) / desired_complex_alpha[i, j]))
                    idx = np.argmin(phase_diffs)
                    discrete_phase[i, j] = valid_discrete_shifts[idx]

            exp_ris = np.exp(1j * discrete_phase)
            bf = np.abs(np.sum(total_path_exp_shift * exp_ris)) ** 2

            if bf > best_bf:
                best_bf = bf
                best_alpha = alpha
                discrete_phase_alpha = discrete_phase.copy()

        print(f"Optimum value of α: {best_alpha:.3f} rad")

        return discrete_phase_alpha

        
class tx_ris_rx_min_phase_diff_opt:
    def __init__(self, tx_antenna: Antenna = None, rx_antenna: Antenna = None, ris: RIS = None, wave_model: str = 'spherical') -> None:
        """
        Parameters:
        -----------
        tx_antenna : Antenna object
            Transmitter antenna object (must be provided).
        rx_antenna : Antenna object
            Receiver antenna object (must be provided).
        ris : RIS object
            Reconfigurable Intelligent Surface object (must be provided).
        wave_model : str
            Wave model to use ('spherical' or 'plane').
        """
        if tx_antenna is None:
            raise ValueError("tx_antenna must be provided")
        if rx_antenna is None:
            raise ValueError("rx_antenna must be provided")
        if ris is None:
            raise ValueError("ris must be provided")

        self.tx_antenna = tx_antenna
        self.rx_antenna = rx_antenna
        self.ris = ris
        self.wave_model = wave_model.lower()

    def optimize(self) -> np.ndarray:
        """
        Compute optimal state (phase shift) for each RIS element. 
        Based on minimizing euclidean phase difference. 
        Defined in Algorithm 1 of Z. Chen, L. X. Cai and X. Hao, 
        "Near-Field and Far-Field Beamforming Design for RIS-enabled 
        Millimeter Wave Systems,", https://ieeexplore.ieee.org/document/10683528

        - mode:
            "cont" => exact inverse phase (continuous phase adjustment).
            "disc" => quantized phase to nearest available RIS state.
        - wave_model:
            "spherical" => phases computed assuming spherical wavefronts.
            "plane"     => phases computed assuming plane wavefronts.
        """
        # Calculate contiunous perfect phase shifts depending on wave model
        total_path_exp_shift = calc_total_path_exp_shift(
            tx_antenna=self.tx_antenna,
            rx_antenna=self.rx_antenna,
            ris=self.ris,
            wave_model=self.wave_model
        )

        # Discretize phase initialization to zeros
        discrete_phase_temp = np.zeros_like(total_path_exp_shift, dtype=float)
        # Euclidean error variable to minimize (e_min)
        e_min = np.inf

        # Valid shifts from dictionary
        valid_discrete_shifts = np.array(list(self.ris.phase_shifts_dict.values()))

        for disc_shift_11 in valid_discrete_shifts:
            # Set first element phase and end-to-end complex term
            discrete_phase_temp[0, 0] = disc_shift_11
            complex_11 = np.exp(1j * disc_shift_11) * total_path_exp_shift[0, 0]
            # initialize e_temp
            e_temp = 0

            # Set remaining elements phases
            for i in range(self.ris.Nrow):
                for j in range(self.ris.Ncol):
                    if i == 0 and j == 0:
                        continue  # Skip first element

                    phase_diffs = np.abs(valid_discrete_shifts - np.angle(complex_11 / total_path_exp_shift[i, j]))
                    idx = np.argmin(phase_diffs)
                    discrete_phase_temp[i, j] = valid_discrete_shifts[idx]

                    # Calculate euclidean error for this configuration
                    complex_ij = np.exp(1j * discrete_phase_temp[i, j]) * total_path_exp_shift[i, j]
                    e_temp += np.linalg.norm(np.angle(complex_ij / complex_11))

            # Update minimum error and best configuration
            if e_temp < e_min:
                e_min = e_temp
                discrete_phase = discrete_phase_temp.copy()

        return discrete_phase
