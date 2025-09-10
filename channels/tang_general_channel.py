import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi

from utils.utils import *
from optimizers.tx_ris_rx_opt import tx_ris_rx_direct_disc_opt, tx_ris_rx_max_bf_opt
from components.antenna import Antenna
from components.ris import RIS

import plotly.io as pio
pio.renderers.default = 'notebook_connected' # 'browser' for avoid error but shown in the browser
import plotly.graph_objects as go

import scienceplots
plt.style.use(['science','no-latex'])
plt.rcParams.update({'font.size': 15})

class tang_general_channel:
    def __init__(self):
        """
        Initialize Tang General Channel Model
        """
        self.model_name = 'Tang General'
        self.last_optimizer = ''

    def optimize(self, optimizer='max_bf', mode='disc', tx_antenna=None, rx_antenna=None, ris=None):
        """
        Optimize the RIS phase configuration given the selected optimizer and system elements.

        Parameters
        ----------
        optimizer : str
            Name of the optimization strategy (e.g., 'direct_disc', 'max_bf').
        mode : str
            Phase configuration mode for the RIS: 
            - 'disc' : discrete phase shifts.
            - 'cont' : continuous phase shifts.
        tx_antenna : Antenna, optional
            Transmitting antenna object (must be of class Antenna).
        rx_antenna : Antenna, optional
            Receiving antenna object (must be of class Antenna).
        ris : RIS, optional
            Reconfigurable Intelligent Surface object (must be of class RIS).
        """

        # Check if tx_antenna, rx_antenna, and ris are set
        # Comprobación de tipos de entrada
        if tx_antenna is not None and not isinstance(tx_antenna, Antenna):
            raise ValueError("tx_antenna must be of class Antenna")
        if rx_antenna is not None and not isinstance(rx_antenna, Antenna):
            raise ValueError("rx_antenna must be of class Antenna")
        if ris is not None and not isinstance(ris, RIS):
            raise ValueError("ris must be of class RIS")
    
        # Check correct mode name
        if mode not in ['disc', 'cont']:
            self.last_optimizer = ''    # If error, reset it
            raise ValueError("mode must be either 'disc' or 'cont'")
        
        if mode == 'disc':
            self.last_optimizer = optimizer     # Update last optimizer used
        else:
            self.last_optimizer = 'perfect'

        # Add extra optimizers as extra elif entries
        if optimizer == 'direct_disc':
            # Instantiate optimizer object
            opt = tx_ris_rx_direct_disc_opt(tx_antenna=tx_antenna, rx_antenna=rx_antenna, ris=ris, mode=mode)
        elif optimizer == 'max_bf':
            # Instantiate optimizer object
            opt = tx_ris_rx_max_bf_opt(tx_antenna=tx_antenna, rx_antenna=rx_antenna, ris=ris, mode=mode)
        else:
            raise ValueError("Invalid optimizer.")
            
        # Optimize the states
        opt_phase_shift = opt.optimize()

        # Update the ris states
        ris.set_all_shift(opt_phase_shift, mode=mode)

    def phase_bf_multrx(self, tx_antenna=None, rx_positions=None, ris=None):
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

        return np.abs(phase_sums)**2
    














    def plot_phase_bf_2d_map(self, tx_antenna=None, ris=None, ax1lim=None, ax2lim=None, Nax1=None, Nax2=None, plane='xz'):
        """
        Plots a 2D heatmap of the normalized phase beamforming pattern over a specified plane.

        This function computes and visualizes the spatial distribution of the beamforming gain
        generated by the RIS across a 2D plane (e.g., XZ, YZ, or XY), normalized with respect 
        to the maximum possible gain. It also overlays the positions of the transmitter (TX), 
        receiver (RX), and RIS panel on the plot for context.

        Parameters
        ----------
        ax1lim : list or tuple of float, shape (2,)
            Limits for the first axis of the plane. `ax1lim[0]` is the lower bound,
            and `ax1lim[1]` is the upper bound.
        
        ax2lim : list or tuple of float, shape (2,)
            Limits for the second axis of the plane. `ax2lim[0]` is the lower bound,
            and `ax2lim[1]` is the upper bound.
        
        Nax1 : int
            Number of sampling points along the first axis (horizontal resolution).
        
        Nax2 : int
            Number of sampling points along the second axis (vertical resolution).
        
        plane : str, optional (default='xz')
            Plane over which the map is computed. Must be a string of two characters 
            from 'x', 'y', or 'z', e.g., 'xz', 'yz', or 'xy'.

        Returns
        -------
        None
            The function displays a 2D color map plot but does not return any value.
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        i1, i2 = axis_map[plane[0]], axis_map[plane[1]]
        fixed_axis = list({0, 1, 2} - {i1, i2})[0]

        ax1 = np.linspace(ax1lim[0], ax1lim[1], Nax1)
        ax2 = np.linspace(ax2lim[0], ax2lim[1], Nax2)
        A1, A2 = np.meshgrid(ax1, ax2, indexing='ij')
        A3 = np.zeros_like(A1)

        rx_points = np.zeros((Nax1 * Nax2, 3))
        rx_points[:, i1] = A1.flatten()
        rx_points[:, i2] = A2.flatten()
        rx_points[:, fixed_axis] = A3.flatten()

        bf = self.phase_bf_multrx(tx_antenna=tx_antenna, rx_positions=rx_points, ris=ris)
        max_bf = (ris.Nrow * ris.Ncol)**2

        plt.figure(figsize=(6, 4))
        pcm = plt.pcolormesh(A1, A2, bf.reshape(Nax1, Nax2) / max_bf, shading='auto', cmap='viridis', vmin=0, vmax=1)
        plt.xlabel(f"{plane[0].upper()} (m)")
        plt.ylabel(f"{plane[1].upper()} (m)")
        plt.title(f"Phase beamforming distribution in {plane.upper()} plane, mode '{ris.mode}', opt '{self.last_optimizer}'")
        plt.colorbar(pcm, label='Normalized Beamforming')
        plt.xlim(ax1lim)
        plt.ylim(ax2lim)

        ax = plt.gca()
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        plt.grid(True)

        tx = tx_antenna.position[0]
        ris_start = ris.positions[0, 0]
        ris_end = ris.positions[-1, -1]

        plt.plot(tx[i1], tx[i2], 'ro', markersize=8)
        plt.text(tx[i1] + 0.3, tx[i2] + 0.3, 'TX', color='red', fontsize=12, ha='left', va='top')

        plt.plot([ris_start[i1], ris_end[i1]], [ris_start[i2], ris_end[i2]], 'r-', linewidth=8)
        plt.text(ris_start[i1] + 0.5, ris_start[i2] + 0.5, 'RIS', color='red', fontsize=12, ha='left', va='top')
        plt.tight_layout()
        plt.show()
    

