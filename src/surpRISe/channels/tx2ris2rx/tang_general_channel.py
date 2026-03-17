import numpy as np
from scipy.constants import c, pi

from surpRISe.channels.tx2ris2rx.tx2ris2rx import tx2ris2rx
from surpRISe.components.antenna import Antenna
from surpRISe.components.ris import RIS
from surpRISe.utils.utils import filt_round_decimals


class tang_general_channel(tx2ris2rx):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "Tang General"

    def phase_bf_multrx(
        self,
        tx_antenna: Antenna = None,
        rx_positions: np.ndarray = None,
        ris: RIS = None,
    ) -> np.ndarray:
        """
        Compute the phase beamforming response at multiple receiver (RX) locations
        considering the transmission path TX → RIS → RX.

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object containing its position attribute.
        rx_positions : ndarray, shape (N, 3)
            Array of receiver positions in 3D coordinates.
        ris : object
            RIS object containing element positions, carrier frequency, and
            element phase shifts.

        Returns
        -------
        phase_bf : ndarray, shape (N,)
            Beamforming power at each receiver location.
        """
        if tx_antenna is None or rx_positions is None or ris is None:
            raise ValueError("tx_antenna, rx_positions, and ris must be provided.")

        rx_positions = np.atleast_2d(rx_positions)
        Nrx = rx_positions.shape[0]

        ris_pos = ris.positions.copy()  # shape: (Nrow, Ncol, 3)
        tx_pos = tx_antenna.position.copy()

        k = 2 * pi * ris.fc / c

        # RIS phase shift exponential contribution
        phase_shift_matrix = ris.elem_phase_shift  # shape: (Nrow, Ncol)
        exp_ris = np.exp(1j * phase_shift_matrix)
        # Avoid float rounding of magnitude greater than 1
        exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)

        phase_sums = np.zeros(Nrx, dtype=complex)

        # Compute distance from TX to each RIS element
        d_tx_ris = np.linalg.norm(ris_pos - tx_pos, axis=2)
        exp_tx_ris = np.exp(-1j * k * d_tx_ris)
        # Avoid float rounding of magnitude greater than 1
        exp_tx_ris = filt_round_decimals(exp_tx_ris, 1, atol=1e-12, rtol=0)

        for i in range(Nrx):
            rx_pos = rx_positions[i]
            d_ris_rx = np.linalg.norm(ris_pos - rx_pos, axis=2)
            exp_ris_rx = np.exp(-1j * k * d_ris_rx)

            # Avoid float rounding of magnitude greater than 1
            exp_ris_rx = filt_round_decimals(exp_ris_rx, 1, atol=1e-12, rtol=0)

            # Element-wise product and sum
            exp_total = exp_tx_ris * exp_ris * exp_ris_rx

            # Avoid float rounding of magnitude greater than 1
            exp_total = filt_round_decimals(exp_total, 1, atol=1e-12, rtol=0)

            phase_sums[i] = np.sum(exp_total)

        return np.abs(phase_sums) ** 2

    def channel_matrix_multrx(
        self,
        tx_antenna: Antenna = None,
        rx_antenna: Antenna = None,
        rx_positions: np.ndarray = None,
        ris: RIS = None,
        f_vector: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute the complex channel coefficients (H) at multiple receiver (RX) locations
        and multiple frequencies considering the path TX → RIS → RX.

        If f_vector is None or a single value, it returns shape (Nrx,).
        If f_vector is a vector, it returns shape (Nrx, Nfreq).

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object.
        rx_antenna : object
            Receiver antenna object.
        rx_positions : ndarray of shape (Nrx, 3)
            3D positions of the receivers.
        ris : object
            RIS object containing physical and operational parameters.
        f_vector : ndarray or list, optional
            Frequency sweep vector in Hz. Defaults to None (uses ris.fc).

        Returns
        -------
        channel_coeffs : ndarray
            Complex channel coefficients (H). 
            Shape is (Nrx,) or (Nrx, Nfreq) depending on input.
        """

        if (
            tx_antenna is None
            or rx_antenna is None
            or rx_positions is None
            or ris is None
        ):
            raise ValueError(
                "tx_antenna, rx_antenna, rx_positions, and ris must be provided."
            )

        # Handle frequency vector logic
        if f_vector is None:
            freqs = np.array([ris.fc])
            is_single_freq = True
        else:
            freqs = np.atleast_1d(f_vector)
            is_single_freq = (freqs.size == 1)

        Nfreq = freqs.size
        k_vector = 2 * np.pi * freqs / c  # Wavenumber vector (Nfreq,)

        # Ensure rx_positions is at least 2D
        rx_positions = np.atleast_2d(rx_positions)
        Nrx = rx_positions.shape[0]

        # Retrieve RIS and antenna information
        ris_pos = ris.positions.copy()  # (Nrow, Ncol, 3)
        ris_center = ris.center.copy()  # (3,)
        tx_pos = tx_antenna.position.copy()  # (1, 3)

        # Antenna gains in linear scale (sqrt for field scaling)
        sqrt_g_tx = np.sqrt(10 ** (tx_antenna.gain_dbi / 10))
        sqrt_g_rx = np.sqrt(10 ** (rx_antenna.gain_dbi / 10))

        # RIS element size and channel constant
        dx, dy = ris.dx, ris.dy
        channel_constant = (sqrt_g_tx * sqrt_g_rx * dx * dy) / (4 * np.pi)

        # RIS reflection coefficient (fixed for design frequency ris.fc)
        # Shape: (Nrow, Ncol)
        exp_ris = ris.get_all_refl_coef()
        exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)

        # Initialize output: (Nrx, Nfreq)
        channel_coeffs = np.zeros((Nrx, Nfreq), dtype=np.complex128)

        # Distance from Tx to RIS (Nrow, Ncol)
        d_tx_ris = np.linalg.norm(ris_pos - tx_pos, axis=2)
        
        # Vectorized phase from Tx to RIS over all frequencies
        # Broadcasting: (Nfreq, 1, 1) * (1, Nrow, Ncol) -> (Nfreq, Nrow, Ncol)
        exp_tx_ris_f = np.exp(-1j * k_vector[:, None, None] * d_tx_ris[None, :, :])
        exp_tx_ris_f = filt_round_decimals(exp_tx_ris_f, 1, atol=1e-12, rtol=0)

        # Pre-calculate Tx radiation pattern toward RIS
        ris_rad_pat_tx = ris.get_rad_pat(tx_pos)
        ris_pos_flat = ris.positions.reshape(-1, 3)
        tx_rad_pat_flat = tx_antenna.get_rad_pat(ris_pos_flat)
        tx_rad_pat = tx_rad_pat_flat.reshape(ris_pos.shape[:2])

        for i in range(Nrx):
            rx_pos = rx_positions[i]

            # Distance and phase from RIS to current Rx
            d_ris_rx = np.linalg.norm(ris_pos - rx_pos, axis=2)
            
            # Vectorized phase RIS to Rx: (Nfreq, Nrow, Ncol)
            exp_ris_rx_f = np.exp(-1j * k_vector[:, None, None] * d_ris_rx[None, :, :])
            exp_ris_rx_f = filt_round_decimals(exp_ris_rx_f, 1, atol=1e-12, rtol=0)

            # Total complex phase response per element and frequency
            # Broadcasting exp_ris to (1, Nrow, Ncol)
            exp_total_f = exp_tx_ris_f * exp_ris[None, :, :] * exp_ris_rx_f
            exp_total_f = filt_round_decimals(exp_total_f, 1, atol=1e-12, rtol=0)

            # Radiation patterns (calculated at design frequency or fixed)
            ris_rad_pat_rx = ris.get_rad_pat(rx_pos)
            
            # RX radiation pattern pointing at RIS center
            pointing_vec_rx = ris_center - rx_pos
            norm_rx = np.linalg.norm(pointing_vec_rx)
            if norm_rx == 0: raise ValueError("Receiver is at RIS center.")
            
            unit_vectors_rx = (ris_pos - rx_pos) / np.linalg.norm(ris_pos - rx_pos, axis=2, keepdims=True)
            cos_ang_rx = np.sum(unit_vectors_rx * (pointing_vec_rx / norm_rx), axis=2)
            ang_rel_rx = np.arccos(np.clip(cos_ang_rx, -1.0, 1.0))

            rx_rad_pat = np.zeros_like(ang_rel_rx)
            valid = (ang_rel_rx >= 0) & (ang_rel_rx <= np.pi / 2)
            gain_linear_rx = 10 ** (rx_antenna.gain_dbi / 10)
            rx_rad_pat[valid] = np.cos(ang_rel_rx[valid]) ** (gain_linear_rx / 2 - 1)

            # Combined patterns in field scale
            f_combine_field = np.sqrt(ris_rad_pat_tx * ris_rad_pat_rx * tx_rad_pat * rx_rad_pat)

            # Sum over RIS elements (axes 1 and 2), preserving frequency axis (0)
            # (Nfreq, Nrow, Ncol) * (1, Nrow, Ncol) / (1, Nrow, Ncol)
            sum_over_elements = np.sum(
                (exp_total_f * f_combine_field[None, :, :]) / (d_tx_ris * d_ris_rx)[None, :, :],
                axis=(1, 2)
            )
            
            channel_coeffs[i, :] = channel_constant * sum_over_elements

        # Return (Nrx,) if it was a single frequency call
        return channel_coeffs.flatten() if is_single_freq else channel_coeffs

    def path_gain_multrx(
        self,
        tx_antenna: Antenna = None,
        rx_antenna: Antenna = None,
        rx_positions: np.ndarray = None,
        ris: RIS = None,
        f_vector: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute the total path gain at multiple receiver (RX) locations
        considering the transmission path TX → RIS → RX.

        The path gain is calculated by combining free-space propagation,
        RIS reflection coefficients, element radiation patterns, and
        antenna gains.

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object containing position and gain attributes.
        rx_antenna : object
            Receiver antenna object containing position and gain attributes.
        rx_positions : ndarray of shape (Nrx, 3)
            Array with the 3D positions of the receivers based on rx_antenna hardware.
        ris : object
            RIS object containing geometry (positions, element spacing, center),
            carrier frequency, reflection coefficients, and radiation patterns.

        Returns
        -------
        path_gains : ndarray of shape (Nrx,)
            Path gain in linear scale at each receiver location.
        """

        return np.abs(self.channel_matrix_multrx(tx_antenna, rx_antenna, rx_positions, ris, f_vector)) ** 2

    def __repr__(self) -> str:
        return "Tang General channel model"
