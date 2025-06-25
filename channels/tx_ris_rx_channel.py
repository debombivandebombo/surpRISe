import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from optimizers.tx_ris_rx_opt import tx_ris_rx_direct_disc_opt, tx_ris_rx_max_bf_opt

class tx_ris_rx_channel:
    def __init__(self, tx_antenna=None, rx_antenna=None, ris=None,
                 mode='normal', model='Tang', optimizer='max_bf'):
        """
        Initializes a RIS-assisted Tx-RIS-Rx channel.

        Parameters:
        -----------
        tx_antenna : Antenna object
            Transmitter antenna object (must be provided).
        rx_antenna : Antenna object
            Receiver antenna object (must be provided).
        ris : RIS object
            Reconfigurable Intelligent Surface object (must be provided).
        mode : str
            Phase configuration mode for RIS ('normal' or 'perfect').
        model : str
            Channel model name (e.g., 'Tang', etc.).
        optimizer : str
            Optimizer name (e.g., 'direct_disc', 'max_bf', etc.).
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
        self.model = model.lower()
        self.optimizer = optimizer.lower()

        if self.mode not in ['normal', 'perfect']:
            raise ValueError("mode must be either 'normal' or 'perfect'")
        
        # Placeholder for RIS-assisted channel computation
        self.h_ris = None           # Complex channel via RIS
        self.optimal_state = None  # Optimal state index matrix (for normal mode)
        self.optimal_shift = None  # Optimal shift matrix (for perfect mode)
    
    def optimize(self):
        """
        Optimize the states/shifts of the RIS given the desired optimizer
        """
        # Add extra optimizers as extra elif entries
        if self.optimizer == 'direct_disc':
            # Instantiate optimizer object
            opt = tx_ris_rx_direct_disc_opt(tx_antenna=self.tx_antenna, rx_antenna=self.rx_antenna, ris=self.ris, mode=self.mode)
        elif self.optimizer == 'max_bf':
            # Instantiate optimizer object
            opt = tx_ris_rx_max_bf_opt(tx_antenna=self.tx_antenna, rx_antenna=self.rx_antenna, ris=self.ris, mode=self.mode)
        else:
            raise ValueError("Invalid optimizer.")
        
        # Optimize the states
        opt_phase_shift = opt.optimize()

        # Update the ris states
        self.ris.set_all_shift(opt_phase_shift, mode=self.mode)

        # Return the updated ris object
        # return self.ris # Not necessary because the original object is already updated

    def phase_bf_multrx(self, rx_positions):
        """
        Compute the phase beamforming at multiple RX locations due to TX-RIS-RX paths.

        Parameters:
        -----------
        rx_positions : ndarray, shape (N, 3)
            Array of receiver positions in 3D.

        Returns:
        --------
        phase_bf : ndarray, shape (N,)
            Beamforming power at each receiver location.
        """
        rx_positions = np.atleast_2d(rx_positions)
        Nrx = rx_positions.shape[0]

        ris_pos = self.ris.positions  # shape: (Nrow, Ncol, 3)
        tx_pos = self.tx_antenna.position
        phase_shift_matrix = self.ris.elem_phase_shift  # shape: (Nrow, Ncol)

        k = 2 * pi * self.ris.fc / c
        phase_sums = np.zeros(Nrx, dtype=complex)

        # Compute distance from TX to each RIS element
        d_tx_ris = np.linalg.norm(ris_pos - tx_pos, axis=2)
        exp_tx_ris = np.exp(-1j * k * d_tx_ris)

        # RIS phase shift contribution
        exp_ris = np.exp(1j * phase_shift_matrix)

        for i in range(Nrx):
            rx_pos = rx_positions[i]
            d_ris_rx = np.linalg.norm(ris_pos - rx_pos, axis=2)
            exp_ris_rx = np.exp(-1j * k * d_ris_rx)

            # Element-wise product and sum
            exp_total = exp_tx_ris * exp_ris * exp_ris_rx
            phase_sums[i] = np.sum(exp_total)

        return np.abs(phase_sums)**2

    def __repr__(self):
        return (f"tx_ris_rx_channel(model='{self.model}', mode='{self.mode}', "
                f"RIS size={self.ris.Nrow}x{self.ris.Ncol})")
