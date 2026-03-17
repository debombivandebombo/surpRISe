import numpy as np
from scipy.constants import c, pi

from surpRISe.channels.tx2ris2rx.tx2ris2rx import tx2ris2rx
from surpRISe.components.antenna import Antenna
from surpRISe.components.ris import RIS
from surpRISe.utils.utils import filt_round_decimals


class tang_ff_channel(tx2ris2rx):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "Tang Farfield"

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
        ris_center = ris.center.copy()  # shape: (3,)
        tx_pos = tx_antenna.position.copy()

        k = 2 * pi * ris.fc / c

        # RIS phase shift exponential contribution
        phase_shift_matrix = ris.elem_phase_shift  # shape: (Nrow, Ncol)
        exp_ris = np.exp(1j * phase_shift_matrix)
        # Avoid float rounding of magnitude greater than 1
        exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)

        phase_sums = np.zeros(Nrx, dtype=complex)

        # Compute TX to RIS phase
        ris_rel_pos = ris_pos - ris_center
        tx_pos = tx_pos.reshape(
            3,
        )
        vec_tx_ris = (ris_center - tx_pos) / np.linalg.norm(
            ris_center - tx_pos
        )  # Unitary vector of TX-RIS direction

        for i in range(Nrx):
            rx_pos = rx_positions[i]
            rx_pos = rx_pos.reshape(
                3,
            )
            vec_ris_rx = (rx_pos - ris_center) / np.linalg.norm(
                rx_pos - ris_center
            )  # Unitary vector of RIS-RX direction

            k_eff = vec_tx_ris - vec_ris_rx

            path_phase_shifts = k * np.tensordot(ris_rel_pos, k_eff, axes=([2], [0]))
            exp_total_path = np.exp(-1j * path_phase_shifts)

            # Avoid float rounding of magnitude greater than 1
            exp_total_path = filt_round_decimals(exp_total_path, 1, atol=1e-12, rtol=0)
            exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)

            exp_total = exp_total_path * exp_ris

            # Avoid float rounding of magnitude greater than 1
            exp_total = filt_round_decimals(exp_total, 1, atol=1e-12, rtol=0)

            phase_sums[i] = np.sum(exp_total)

        # # Compute TX to RIS phase
        # ris_rel_pos = ris_pos - ris_center
        # tx_pos = tx_pos.reshape(3,)
        # vec_tx_ris = (ris_center - tx_pos) / np.linalg.norm(ris_center - tx_pos) # Unitary vector of TX-RIS direction
        # dot_product_tx = np.tensordot(ris_rel_pos, vec_tx_ris, axes=([2], [0])) # (Nh, Nv)

        # d_tx_ris_center = np.linalg.norm(ris_center - tx_pos)
        # exp_tx_ris_center = np.exp(-1j * k * d_tx_ris_center) # (Nh, Nv)
        # exp_tx_ris = np.exp(-1j * k * dot_product_tx) * exp_tx_ris_center # (Nh, Nv)

        # for i in range(Nrx):
        #     rx_pos = rx_positions[i]
        #     rx_pos = rx_pos.reshape(3,)
        #     vec_ris_rx = (rx_pos - ris_center) / np.linalg.norm(rx_pos - ris_center) # Unitary vector of RIS-RX direction
        #     dot_product_rx = np.tensordot(ris_rel_pos, vec_ris_rx, axes=([2], [0])) # (Nh, Nv)

        #     d_ris_center_rx = np.linalg.norm(rx_pos - ris_center)
        #     exp_ris_center_rx = np.exp(-1j * k * d_ris_center_rx) # (Nh, Nv)
        #     exp_ris_rx = np.exp(1j * k * dot_product_rx) * exp_ris_center_rx # (Nh, Nv)

        #     # Avoid float rounding of magnitude greater than 1
        #     exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)
        #     exp_tx_ris = filt_round_decimals(exp_tx_ris, 1, atol=1e-12, rtol=0)
        #     exp_ris_rx = filt_round_decimals(exp_ris_rx, 1, atol=1e-12, rtol=0)

        #     # Element-wise product and sum
        #     exp_total = exp_tx_ris * exp_ris * exp_ris_rx

        #     # Avoid float rounding of magnitude greater than 1
        #     exp_total = filt_round_decimals(exp_total, 1, atol=1e-12, rtol=0)

        #     phase_sums[i] = np.sum(exp_total)

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
        and multiple frequencies considering the path TX → RIS → RX under Far-Field assumption.
        """

        if tx_antenna is None or rx_antenna is None or rx_positions is None or ris is None:
            raise ValueError("tx_antenna, rx_antenna, rx_positions, and ris must be provided.")

        if f_vector is None:
            freqs = np.array([ris.fc])
            is_single_freq = True
        else:
            freqs = np.atleast_1d(f_vector)
            is_single_freq = (freqs.size == 1)

        Nfreq = freqs.size
        k_vector = 2 * np.pi * freqs / c  # (Nfreq,)

        rx_positions = np.atleast_2d(rx_positions)
        Nrx = rx_positions.shape[0]

        ris_pos = ris.positions.copy()  # (Nrow, Ncol, 3)
        ris_center = ris.center.copy()  # (3,)
        tx_pos = tx_antenna.position.copy().flatten()  # (3,)

        sqrt_g_tx = np.sqrt(10 ** (tx_antenna.gain_dbi / 10))
        sqrt_g_rx = np.sqrt(10 ** (rx_antenna.gain_dbi / 10))

        dx, dy = ris.dx, ris.dy
        channel_constant = (sqrt_g_tx * sqrt_g_rx * dx * dy) / (4 * np.pi)

        exp_ris = ris.get_all_refl_coef()
        exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)

        channel_coeffs = np.zeros((Nrx, Nfreq), dtype=np.complex128)

        ris_rel_pos = ris_pos - ris_center

        vec_tx_ris = (ris_center - tx_pos) / np.linalg.norm(ris_center - tx_pos)
        dot_product_tx = np.tensordot(ris_rel_pos, vec_tx_ris, axes=([2], [0]))  # (Nrow, Ncol)

        d_tx_ris_center = np.linalg.norm(ris_center - tx_pos)
        
        # Phase from TX to RIS
        exp_tx_ris_f = np.exp(-1j * k_vector[:, None, None] * (d_tx_ris_center + dot_product_tx)[None, :, :])
        exp_tx_ris_f = filt_round_decimals(exp_tx_ris_f, 1, atol=1e-12, rtol=0)

        # Radiation pattern toward RIS (from RIS center)
        cos_ang_tx = np.sum(-vec_tx_ris * ris.normal)
        cos_ang_tx = np.clip(cos_ang_tx, -1.0, 1.0)
        ang_rel_tx = np.arccos(cos_ang_tx)
        ris_rad_pat_tx = np.cos(ang_rel_tx) ** ris.radiation_factor if (0 <= ang_rel_tx <= np.pi / 2) else 0

        # TX intrinsic radiation pattern
        tx_rad_pat_flat = tx_antenna.get_rad_pat(ris_pos.reshape(-1, 3))
        tx_rad_pat = tx_rad_pat_flat.reshape(ris_pos.shape[:2])

        for i in range(Nrx):
            rx_pos = rx_positions[i].flatten()

            vec_ris_rx = (rx_pos - ris_center) / np.linalg.norm(rx_pos - ris_center)
            dot_product_rx = np.tensordot(ris_rel_pos, vec_ris_rx, axes=([2], [0]))  # (Nrow, Ncol)

            d_ris_center_rx = np.linalg.norm(rx_pos - ris_center)
            
            # Phase from RIS to RX
            exp_ris_rx_f = np.exp(-1j * k_vector[:, None, None] * (d_ris_center_rx - dot_product_rx)[None, :, :])
            exp_ris_rx_f = filt_round_decimals(exp_ris_rx_f, 1, atol=1e-12, rtol=0)

            exp_total_f = exp_tx_ris_f * exp_ris[None, :, :] * exp_ris_rx_f
            exp_total_f = filt_round_decimals(exp_total_f, 1, atol=1e-12, rtol=0)

            cos_ang_rx = np.sum(vec_ris_rx * ris.normal)
            cos_ang_rx = np.clip(cos_ang_rx, -1.0, 1.0)
            ang_rel_rx = np.arccos(cos_ang_rx)
            ris_rad_pat_rx = np.cos(ang_rel_rx) ** ris.radiation_factor if (0 <= ang_rel_rx <= np.pi / 2) else 0

            # Rx antenna radiation pattern
            pointing_vec_rx = ris_center - rx_pos
            norm_rx = np.linalg.norm(pointing_vec_rx)
            if norm_rx == 0: raise ValueError("Receiver is at RIS center.")
            
            vec_rx_ris = pointing_vec_rx / norm_rx
            unit_vectors_rx = (ris_pos - rx_pos) / np.linalg.norm(ris_pos - rx_pos, axis=2, keepdims=True)
            cos_ang_arx = np.sum(unit_vectors_rx * vec_rx_ris, axis=2)
            ang_rel_arx = np.arccos(np.clip(cos_ang_arx, -1.0, 1.0))
            
            rx_rad_pat = np.zeros_like(ang_rel_arx)
            valid = (ang_rel_arx >= 0) & (ang_rel_arx <= np.pi / 2)
            rx_rad_pat[valid] = np.cos(ang_rel_arx[valid]) ** ((10 ** (rx_antenna.gain_dbi / 10)) / 2 - 1)

            f_combine_field = np.sqrt(ris_rad_pat_tx * ris_rad_pat_rx * tx_rad_pat * rx_rad_pat)

            # In far-field, amplitude decay uses center distances
            amp_decay = d_tx_ris_center * d_ris_center_rx

            sum_over_elements = np.sum((exp_total_f * f_combine_field[None, :, :]) / amp_decay, axis=(1, 2))
            channel_coeffs[i, :] = channel_constant * sum_over_elements

        return channel_coeffs.flatten() if is_single_freq else channel_coeffs


    def path_gain_multrx(
        self,
        tx_antenna: Antenna = None,
        rx_antenna: Antenna = None,
        rx_positions: np.ndarray = None,
        ris: RIS = None,
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

        if (
            tx_antenna is None
            or rx_antenna is None
            or rx_positions is None
            or ris is None
        ):
            raise ValueError(
                "tx_antenna, rx_antenna, rx_positions, and ris must be provided."
            )

        # Ensure rx_positions is at least 2D
        rx_positions = np.atleast_2d(rx_positions)
        Nrx = rx_positions.shape[0]

        # Retrieve RIS and antenna information
        ris_pos = ris.positions.copy()  # shape: (Nrow, Ncol, 3)
        ris_center = ris.center.copy()  # shape: (3,)
        tx_pos = tx_antenna.position.copy()  # shape: (1, 3)

        # Wavenumber
        k = 2 * pi * ris.fc / c

        # Antenna gains in linear scale
        g_tx = 10 ** (tx_antenna.gain_dbi / 10)
        g_rx = 10 ** (rx_antenna.gain_dbi / 10)

        # RIS element size
        dx = ris.dx
        dy = ris.dy

        # Path gain constant
        gain_constant = g_tx * g_rx * (dx * dy) ** 2 / (16 * pi**2)

        # RIS reflection coefficient (Nrow x Ncol complex matrix)
        exp_ris = ris.get_all_refl_coef()

        # Avoid float rounding of magnitude greater than 1
        exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)

        # Placeholder for path gains per receiver
        path_gains = np.zeros(Nrx)

        # TX - RIS center distance
        tx_pos = tx_pos.reshape(
            3,
        )
        d_tx_ris_center = np.linalg.norm(ris_center - tx_pos)

        # Obtain RIS rad pat for TX respect to RIS center
        dir_vec_tx = (tx_pos - ris_center) / np.linalg.norm(
            tx_pos - ris_center
        )  # shape: (3, )
        cos_ang = np.sum(dir_vec_tx * ris.normal)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)  # Ensure valid range
        ang_rel = np.arccos(cos_ang)

        # Compute radiation pattern
        rad_pat_tx = 0
        if (ang_rel >= 0) & (ang_rel <= np.pi / 2):  # valid angle
            rad_pat_tx = np.cos(ang_rel) ** ris.radiation_factor

        gain_constant = (
            gain_constant * rad_pat_tx / (d_tx_ris_center**2)
        )  # TX common factors added

        # Compute TX to RIS phase
        ris_rel_pos = ris_pos - ris_center
        tx_pos = tx_pos.reshape(
            3,
        )
        vec_tx_ris = (ris_center - tx_pos) / np.linalg.norm(
            ris_center - tx_pos
        )  # Unitary vector of TX-RIS direction
        dot_product_tx = np.tensordot(
            ris_rel_pos, vec_tx_ris, axes=([2], [0])
        )  # (Nh, Nv)

        d_tx_ris_center = np.linalg.norm(ris_center - tx_pos)
        exp_tx_ris_center = np.exp(-1j * k * d_tx_ris_center)  # (Nh, Nv)
        exp_tx_ris = np.exp(-1j * k * dot_product_tx) * exp_tx_ris_center  # (Nh, Nv)

        # Avoid float rounding of magnitude greater than 1
        exp_tx_ris = filt_round_decimals(exp_tx_ris, 1, atol=1e-12, rtol=0)

        for i in range(Nrx):
            rx_pos = rx_positions[i]
            rx_pos = rx_pos.reshape(
                3,
            )

            # RIS center - RX distance
            d_ris_center_rx = np.linalg.norm(rx_pos - ris_center)

            # Obtain RIS rad pat for TX respect to RIS center
            dir_vec_rx = (rx_pos - ris_center) / np.linalg.norm(
                rx_pos - ris_center
            )  # shape: (3, )
            cos_ang = np.sum(dir_vec_rx * ris.normal)
            cos_ang = np.clip(cos_ang, -1.0, 1.0)  # Ensure valid range
            ang_rel = np.arccos(cos_ang)

            # Compute radiation pattern
            rad_pat_rx = 0
            if (ang_rel >= 0) & (ang_rel <= np.pi / 2):  # valid angle
                rad_pat_rx = np.cos(ang_rel) ** ris.radiation_factor

            gain_constant_rx = (
                gain_constant * rad_pat_rx / (d_ris_center_rx**2)
            )  # RX common factors added

            # Compute RIS - RX phase
            vec_ris_rx = (rx_pos - ris_center) / np.linalg.norm(
                rx_pos - ris_center
            )  # Unitary vector of RIS-RX direction
            dot_product_rx = np.tensordot(
                ris_rel_pos, vec_ris_rx, axes=([2], [0])
            )  # (Nh, Nv)

            exp_ris_center_rx = np.exp(-1j * k * d_ris_center_rx)  # (Nh, Nv)
            exp_ris_rx = np.exp(1j * k * dot_product_rx) * exp_ris_center_rx  # (Nh, Nv)

            # Avoid float rounding of magnitude greater than 1
            exp_ris_rx = filt_round_decimals(exp_ris_rx, 1, atol=1e-12, rtol=0)

            exp_total = exp_tx_ris * exp_ris * exp_ris_rx

            # Avoid float rounding of magnitude greater than 1
            exp_total = filt_round_decimals(exp_total, 1, atol=1e-12, rtol=0)

            # Path gain computation
            path_gains[i] = gain_constant_rx * np.abs(np.sum(exp_total)) ** 2

        return path_gains

    def __repr__(self) -> str:
        return "Tang Farfield channel model"
