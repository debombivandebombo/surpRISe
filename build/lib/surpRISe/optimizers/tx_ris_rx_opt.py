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

def quantize_phases(self, ideal_phases: np.ndarray, valid_discrete_shifts: np.ndarray) -> np.ndarray:
    """
    Quantize ideal phases to nearest valid discrete shifts.
    Parameters:
    -----------
    ideal_phases : np.ndarray
        Ideal continuous phases to be quantized.
    valid_discrete_shifts : np.ndarray
        Array of valid discrete phase shifts.
    """
    # Expand dimensions for broadcasting
    ideal_phases_expanded = np.expand_dims(ideal_phases, axis=-1)
    shifts_expanded = valid_discrete_shifts[np.newaxis, np.newaxis, :]
    
    # Calculate phase differences and find indices of minimum
    phase_diffs = (ideal_phases_expanded - shifts_expanded + np.pi) % (2 * np.pi) - np.pi # Wrap to [-π, π]
    idx_min = np.argmin(np.abs(phase_diffs), axis=-1)
    
    return valid_discrete_shifts[idx_min]

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
        Based on direct discretization of continuous optimal phases.
        """
        # Obtain continuous perfect phase shifts depending on wave model (for alpha=0)
        desired_complex, desired_phase = calc_tx_ris_rx_cont_opt(
            tx_antenna=self.tx_antenna, 
            rx_antenna=self.rx_antenna, 
            ris=self.ris, 
            wave_model=self.wave_model
        )

        # Obtain valid discrete phase shifts from RIS dictionary
        valid_discrete_shifts = np.array(sorted(list(self.ris.phase_shifts_dict.values())))
        
        # Call the global vectorized quantization function
        discrete_phase = quantize_phases(
            ideal_phases=desired_phase,
            valid_discrete_shifts=valid_discrete_shifts
        )
        
        return discrete_phase
            
class tx_ris_rx_max_bf_grid_search_opt:
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

    def optimize(self, return_history: bool = False) -> np.ndarray:
        """
        Compute optimal state (phase shift) for each RIS element.
        Grid search over alpha in [0, 2pi) to maximize beamforming gain.
        """

        # Obtain continuous perfect phase shifts depending on wave model (for alpha=0)
        desired_complex, desired_phase = calc_tx_ris_rx_cont_opt(
            tx_antenna=self.tx_antenna, 
            rx_antenna=self.rx_antenna, 
            ris=self.ris, 
            wave_model=self.wave_model
        )

        # Obtain channel matrix and discrete phase shifts
        total_path_exp_shift = desired_complex ** -1
        valid_discrete_shifts = np.array(sorted(list(self.ris.phase_shifts_dict.values())))

        # Grid search
        alphas = np.linspace(0, 2 * np.pi, 360)
        best_bf = -np.inf
        best_alpha = 0
        discrete_phase_alpha = np.zeros_like(desired_phase, dtype=float)
        
        if return_history:
            history = {"alphas": [], "gains": []}

        for alpha in alphas:
            # Obtain ideal phases for this 'alpha'
            # (Adding 'alpha' rotates the ideal phases)
            ideal_phases_for_alpha = desired_phase + alpha
            
            # Call the global vectorized quantization function
            discrete_phase = quantize_phases(
                ideal_phases=ideal_phases_for_alpha,
                valid_discrete_shifts=valid_discrete_shifts
            )

            # Calculate beamforming gain (BF) for this configuration
            exp_ris = np.exp(1j * discrete_phase)
            bf = np.abs(np.sum(total_path_exp_shift * exp_ris)) ** 2
            
            if return_history:
                history["alphas"].append(alpha)
                history["gains"].append(bf)

            # Update if this is the best found so far
            if bf > best_bf:
                best_bf = bf
                best_alpha = alpha
                discrete_phase_alpha = discrete_phase.copy()

        print(f"Optimum value of α: {best_alpha:.3f} rad")

        if return_history:
            history["alphas"] = np.array(history["alphas"])
            history["gains"] = np.array(history["gains"])
            return discrete_phase_alpha, history
        else:
            return discrete_phase_alpha

        
class tx_ris_rx_chen_alg1_opt:
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
        """
        # Calculate total path exponential shift
        total_path_exp_shift = calc_total_path_exp_shift(
            tx_antenna=self.tx_antenna,
            rx_antenna=self.rx_antenna,
            ris=self.ris,
            wave_model=self.wave_model
        )

        # Initialize variables
        e_min = np.inf
        valid_discrete_shifts = np.array(sorted(list(self.ris.phase_shifts_dict.values())))
        # Initialize the array that will contain the best configuration
        discrete_phase = np.zeros_like(total_path_exp_shift, dtype=float)

        # 3. Outer loop: Try each discrete phase for element (0, 0)
        for disc_shift_11 in valid_discrete_shifts:
            # Set the reference complex value based on element (0, 0)
            complex_11 = np.exp(1j * disc_shift_11) * total_path_exp_shift[0, 0]

            # Calculate the ideal phases (matrix) for all elements
            #     The ideal phase for element (i,j) is the one that makes
            #     its total phase match that of element (0,0).
            #     ideal_phase_ij = angle(complex_11 / total_path_exp_shift[i, j])
            ideal_phases_matrix = np.angle(complex_11 / total_path_exp_shift)
            
            # Quantize ALL elements at once
            #     (Note: Element (0,0) will be correctly quantized to disc_shift_11
            #     because that is its ideal phase and it is in valid_discrete_shifts)
            discrete_phase_temp = quantize_phases(
                ideal_phases=ideal_phases_matrix,
                valid_discrete_shifts=valid_discrete_shifts
            )

            # Calculate the Euclidean error (sum of absolute phase differences)
            complex_matrix = np.exp(1j * discrete_phase_temp) * total_path_exp_shift
            angle_diffs = np.angle(complex_matrix / complex_11) # Phase differences [-pi, pi]
            e_temp = np.sum(np.abs(angle_diffs))

            # Update minimum error and best configuration
            if e_temp < e_min:
                e_min = e_temp
                discrete_phase = discrete_phase_temp.copy()

        return discrete_phase
    
class tx_ris_rx_max_bf_iterative_opt:
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
        if tx_antenna is None: raise ValueError("tx_antenna must be provided")
        if rx_antenna is None: raise ValueError("rx_antenna must be provided")
        if ris is None: raise ValueError("ris must be provided")
        self.tx_antenna, self.rx_antenna, self.ris, self.wave_model = tx_antenna, rx_antenna, ris, wave_model.lower()
        self.N = self.ris.Nrow * self.ris.Ncol

    def optimize(self, max_iterations: int = 50, tolerance: float = 1e-6, return_history: bool = False):
        """
        Compute optimal state (phase shift) for each RIS element.
        Iterative optimization of alpha to maximize beamforming gain.
        """
        # Compute continuous perfect phase shifts depending on wave model
        desired_complex, desired_phase_ideal_0 = calc_tx_ris_rx_cont_opt(
            tx_antenna=self.tx_antenna, rx_antenna=self.rx_antenna, ris=self.ris, wave_model=self.wave_model)
        total_path_complex_shift = desired_complex ** -1
        valid_discrete_shifts = np.array(sorted(list(self.ris.phase_shifts_dict.values())))

        if return_history:
            history = {"gains": [], "alphas": []}
        
        # Initialization and iterative optimization
        iteration, current_alpha = 0, 0.0
        discrete_phase = np.zeros_like(desired_phase_ideal_0, dtype=float)

        while iteration < max_iterations:
            # Update ideal phases with current alpha
            ideal_phases_for_alpha = (current_alpha + desired_phase_ideal_0)
            # Quantize phases
            discrete_phase = self.quantize_phases(ideal_phases_for_alpha, valid_discrete_shifts)
            # Compute new alpha
            exp_ris = np.exp(1j * discrete_phase)
            y_vector = np.sum(total_path_complex_shift * exp_ris)
            new_alpha = np.angle(y_vector)
            
            if return_history:
                history["alphas"].append(current_alpha)
                history["gains"].append(np.abs(y_vector)**2)

            # Check convergence
            diff = (new_alpha - current_alpha + np.pi) % (2 * np.pi) - np.pi
            if np.abs(diff) < tolerance:
                print(f"Iterative solution converged in {iteration + 1} iterations.")
                break
            
            # Update for next iteration
            current_alpha, iteration = new_alpha, iteration + 1
        
        if iteration == max_iterations:
            print(f"Iterative solution reached max iterations ({max_iterations}).")

        print(f"Optimum value of α found: {current_alpha:.3f} rad")

        if return_history:
            history["iteration"] = np.arange(len(history["gains"]))
            history["gains"], history["alphas"] = np.array(history["gains"]), np.array(history["alphas"])
            return discrete_phase, history
        else:
            return discrete_phase
        
class tx_ris_rx_max_bf_1D_search_opt:
    def __init__(self, tx_antenna: 'Antenna' = None, rx_antenna: 'Antenna' = None, ris: 'RIS' = None, wave_model: str = 'spherical') -> None:
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
        if tx_antenna is None: raise ValueError("tx_antenna must be provided")
        if rx_antenna is None: raise ValueError("rx_antenna must be provided")
        if ris is None: raise ValueError("ris must be provided")
        self.tx_antenna, self.rx_antenna, self.ris, self.wave_model = tx_antenna, rx_antenna, ris, wave_model.lower()
        self.N = self.ris.Nrow * self.ris.Ncol

    def _get_candidate_alphas(self, channel_phases_flat: np.ndarray, valid_discrete_shifts: np.ndarray) -> np.ndarray:
        # Calculate decision boundaries
        M = len(valid_discrete_shifts)
        decision_boundaries = np.zeros(M)
        for k in range(M):
            p_k, p_k_next = valid_discrete_shifts[k], valid_discrete_shifts[(k + 1) % M]
            if p_k_next <= p_k: p_k_next += 2 * np.pi
            decision_boundaries[k] = (p_k + p_k_next) / 2.0
        candidate_alphas = [ (b_k + phi_i) % (2 * np.pi) for i in range(self.N) for k in range(M) for phi_i in [channel_phases_flat[i]] for b_k in decision_boundaries ]
        candidate_alphas.append(0.0)
        
        return np.array(list(set(candidate_alphas)))

    def optimize(self, return_history: bool = False):
        """
        Compute optimal state (phase shift) for each RIS element.
        1D Search over alpha to maximize beamforming gain.
        Only alphas that change the quantized phase configuration are tested.
        """
        # Compute continuous perfect phase shifts depending on wave model
        desired_complex, desired_phase_ideal_0 = calc_tx_ris_rx_cont_opt(
            tx_antenna=self.tx_antenna, rx_antenna=self.rx_antenna, ris=self.ris, wave_model=self.wave_model)
        
        total_path_complex_shift = desired_complex ** -1
        channel_phases = np.angle(total_path_complex_shift)
        valid_discrete_shifts = np.array(sorted(list(self.ris.phase_shifts_dict.values())))
        
        # Determine candidate alphas that change the quantized phase configuration
        alphas_to_test = self._get_candidate_alphas(channel_phases.flatten(), valid_discrete_shifts)
        
        best_bf, best_alpha, best_discrete_phase = -np.inf, 0, np.zeros_like(channel_phases, dtype=float)
        
        if return_history:
            history = {"alphas": [], "gains": []}

        # Evaluate beamforming gain for each candidate alpha
        for alpha in alphas_to_test:
            # Update ideal phases with current alpha
            ideal_phases_for_alpha = (alpha + desired_phase_ideal_0)
            # Quantize phases
            discrete_phase = self.quantize_phases(ideal_phases_for_alpha, valid_discrete_shifts)
            # Compute beamforming gain
            exp_ris = np.exp(1j * discrete_phase)
            bf = np.abs(np.sum(total_path_complex_shift * exp_ris)) ** 2
            # Update best solution if improved
            if bf > best_bf:
                best_bf, best_alpha, best_discrete_phase = bf, alpha, discrete_phase.copy()
            
            if return_history:
                history["alphas"].append(alpha)
                history["gains"].append(bf)

        print(f"1D Search: Trying {len(alphas_to_test)} angles. Optimal alpha={best_alpha:.3f} rad")
        
        if return_history:
            history["alphas"], history["gains"] = np.array(history["alphas"]), np.array(history["gains"])
            return best_discrete_phase, history
        else:
            return best_discrete_phase
