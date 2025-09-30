import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from scipy.optimize import minimize_scalar

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

def calc_plane_path_exp_shift(tx_pos: np.ndarray, rx_pos: np.ndarray, ris: 'RIS', fc: float) -> np.ndarray:
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

class tx_ris_rx_direct_disc_opt:
    def __init__(self, tx_antenna: 'Antenna' = None, rx_antenna: 'Antenna' = None, ris: 'RIS' = None, mode: str = 'disc', wave_model: str = 'spherical') -> None:
        """
        Parameters:
        -----------
        tx_antenna : Antenna object
            Transmitter antenna object (must be provided).
        rx_antenna : Antenna object
            Receiver antenna object (must be provided).
        ris : RIS object
            Reconfigurable Intelligent Surface object (must be provided).
        mode : str
            Phase configuration mode for RIS ('disc' or 'cont').
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

        self.mode = mode.lower()
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
        # Compute total path exponential shift depending on wave model
        if self.wave_model == "spherical":
            total_path_exp_shift = calc_spherical_path_exp_shift(
                self.tx_antenna.position,
                self.rx_antenna.position,
                self.ris.positions,
                self.ris.fc
            )

        elif self.wave_model == "plane":
            total_path_exp_shift = calc_plane_path_exp_shift(
                self.tx_antenna.position,
                self.rx_antenna.position,
                self.ris,
                self.ris.fc
            )

        else:
            raise ValueError("Invalid wave_model. Use 'spherical' or 'plane'.")
        
        # Calculate cont (perfect) phase shift
        desired_complex = total_path_exp_shift ** -1
        desired_phase = np.angle(desired_complex)

        if self.mode == "cont":
            return desired_phase

        elif self.mode == "disc":
            # Discretize phase to nearest value in phase_shifts_dict
            discrete_phase = np.zeros_like(desired_phase)
            # Valid shifts from dictionary
            valid_discrete_shifts = np.array(list(self.ris.phase_shifts_dict.values()))
            for i in range(self.ris.Nrow):
                for j in range(self.ris.Ncol):
                    phase_diffs = np.abs(np.angle(np.exp(1j * valid_discrete_shifts) / desired_complex[i, j]))
                    idx = np.argmin(phase_diffs)
                    state_key = list(self.ris.phase_shifts_dict.keys())[idx]
                    discrete_phase[i, j] = self.ris.phase_shifts_dict[state_key]
            return discrete_phase

        else:
            raise ValueError("Invalid mode. Use 'disc' or 'cont'.")
            
class tx_ris_rx_max_bf_opt:
    def __init__(self, tx_antenna: 'Antenna' = None, rx_antenna: 'Antenna' = None, ris: 'RIS' = None, mode: str = 'disc') -> None:
        """
        Parameters:
        -----------
        tx_antenna : Antenna object
            Transmitter antenna object (must be provided).
        rx_antenna : Antenna object
            Receiver antenna object (must be provided).
        ris : RIS object
            Reconfigurable Intelligent Surface object (must be provided).
        mode : str
            Phase configuration mode for RIS ('disc' or 'cont').
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

        self.mode = mode.lower()

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
        # Compute total path exponential shift depending on wave model
        if self.wave_model == "spherical":
            total_path_exp_shift = calc_spherical_path_exp_shift(
                self.tx_antenna.position,
                self.rx_antenna.position,
                self.ris.positions,
                self.ris.fc
            )

        elif self.wave_model == "plane":
            total_path_exp_shift = calc_plane_path_exp_shift(
                self.tx_antenna.position,
                self.rx_antenna.position,
                self.ris,
                self.ris.fc
            )

        else:
            raise ValueError("Invalid wave_model. Use 'spherical' or 'plane'.")
        
        # Calculate cont (perfect) phase shift
        desired_complex = total_path_exp_shift ** -1
        desired_phase = np.angle(desired_complex)

        if self.mode == "cont":
            return desired_phase

        elif self.mode == "disc":
            valid_discrete_shifts = np.array(list(self.ris.phase_shifts_dict.values()))

            # Minimization algorithm (does not work correctly):
            # def objective(alpha):
            #     # Adjusted phase with alpha
            #     desired_complex_alpha = desired_complex * np.exp(1j * alpha)

            #     # Initialize discrete phases matrix
            #     discrete_phase = np.zeros_like(desired_phase)

            #     # Discretize each element
            #     for i in range(self.ris.Nrow):
            #         for j in range(self.ris.Ncol):
            #             phase_diffs = np.abs(np.angle(np.exp(1j * valid_discrete_shifts) / desired_complex_alpha[i, j]))
            #             idx = np.argmin(phase_diffs)
            #             discrete_phase[i, j] = valid_discrete_shifts[idx]

            #     # Obtain exponential form of the discrete phases
            #     exp_ris = np.exp(1j * discrete_phase)

            #     # Calculate beamforming
            #     bf = np.abs(np.sum(total_path_exp_shift * exp_ris)) ** 2

            #     return -bf  # Minimize the neagitve Beamforming

            # # Optimize in the interval [0, 2π]
            # result = minimize_scalar(objective, bounds=(0, 2 * np.pi), method='bounded')
            # alpha_opt = result.x

            # Grid search
            alphas = np.linspace(0, 2 * np.pi, 720)
            best_bf = -np.inf
            best_alpha = 0

            for alpha in alphas:
                desired_complex_alpha = desired_complex * np.exp(1j * alpha)
                discrete_phase = np.zeros_like(desired_phase)

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

            alpha_opt = best_alpha

            print(f"Optimum value of α: {alpha_opt:.3f} rad")

            # Adjusted phase with optimal alpha
            desired_complex_alpha = desired_complex * np.exp(1j * alpha_opt)

            # Initialize discrete phases matrix
            discrete_phase = np.zeros_like(desired_phase)

            # Discretize each element
            for i in range(self.ris.Nrow):
                for j in range(self.ris.Ncol):
                    phase_diffs = np.abs(np.angle(np.exp(1j * valid_discrete_shifts) / desired_complex_alpha[i, j]))
                    idx = np.argmin(phase_diffs)
                    discrete_phase[i, j] = valid_discrete_shifts[idx]

            return discrete_phase

        else:
            raise ValueError("Invalid mode. Use 'disc' or 'cont'.")