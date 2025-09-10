import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi

from utils.utils import *

import plotly.io as pio
pio.renderers.default = 'notebook_connected' # 'browser' for avoid error but shown in the browser
import plotly.graph_objects as go

from optimizers.tx_ris_rx_opt import tx_ris_rx_direct_disc_opt, tx_ris_rx_max_bf_opt

import scienceplots
plt.style.use(['science','no-latex'])
plt.rcParams.update({'font.size': 15})

class tx_ris_rx_channel:
    def __init__(self, tx_antenna=None, rx_antenna=None, ris=None,
                 model='Tang_general'):
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
        model : str
            Channel model name (e.g., 'Tang_general', 'Tang_farfield', etc.).
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
        self.model = model.lower()
        self.last_optimizer = ''

        # Validate model
        if self.model not in ['tang_general', 'tang_farfield']:
            raise ValueError("Invalid model. Supported models: 'Tang_general', 'Tang_farfield'.")
    
    def optimize(self, optimizer='max_bf', mode='disc'):
        """
        Optimize the states/shifts of the RIS given the desired optimizer

        Parameters:
        -----------
        optimizer : str
            Optimizer name (e.g., 'direct_disc', 'max_bf', etc.).
        mode : str
            Phase configuration mode for RIS ('disc' (discrete) or 'cont' (continuous)).
        """
        # Check correct mode name
        if mode not in ['disc', 'cont']:
            self.last_optimizer = ''    # If error, reset it
            raise ValueError("mode variable must be either 'disc' or 'cont'")
        
        if mode == 'disc':
            self.last_optimizer = optimizer     # Update last optimizer used
        else:
            self.last_optimizer = 'perfect'

        # Add extra optimizers as extra elif entries
        if optimizer == 'direct_disc':
            # Instantiate optimizer object
            opt = tx_ris_rx_direct_disc_opt(tx_antenna=self.tx_antenna, rx_antenna=self.rx_antenna, ris=self.ris, mode=mode)
        elif optimizer == 'max_bf':
            # Instantiate optimizer object
            opt = tx_ris_rx_max_bf_opt(tx_antenna=self.tx_antenna, rx_antenna=self.rx_antenna, ris=self.ris, mode=mode)
        else:
            raise ValueError("Invalid optimizer.")
            
        # Optimize the states
        opt_phase_shift = opt.optimize()

        # Update the ris states
        self.ris.set_all_shift(opt_phase_shift, mode=mode)

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
        def compute_bf_for_rx(exp_ris, exp_tx_ris, exp_ris_rx):
            # Avoid float rounding of magnitude greater than 1
            exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)
            exp_tx_ris = filt_round_decimals(exp_tx_ris, 1, atol=1e-12, rtol=0)
            exp_ris_rx = filt_round_decimals(exp_ris_rx, 1, atol=1e-12, rtol=0)

            # Element-wise product and sum
            exp_total = exp_tx_ris * exp_ris * exp_ris_rx

            # Avoid float rounding of magnitude greater than 1
            exp_total = filt_round_decimals(exp_total, 1, atol=1e-12, rtol=0)

            return np.sum(exp_total)

        rx_positions = np.atleast_2d(rx_positions)
        Nrx = rx_positions.shape[0]

        ris_pos = self.ris.positions.copy()  # shape: (Nrow, Ncol, 3)
        ris_center = self.ris.center.copy()  # shape: (3,)
        tx_pos = self.tx_antenna.position.copy()

        k = 2 * pi * self.ris.fc / c

        # RIS phase shift exponential contribution
        phase_shift_matrix = self.ris.elem_phase_shift  # shape: (Nrow, Ncol)
        exp_ris = np.exp(1j * phase_shift_matrix)
        
        phase_sums = np.zeros(Nrx, dtype=complex)

        if self.model == 'tang_general':    
            # Compute distance from TX to each RIS element
            d_tx_ris = np.linalg.norm(ris_pos - tx_pos, axis=2)
            exp_tx_ris = np.exp(-1j * k * d_tx_ris)

            for i in range(Nrx):
                rx_pos = rx_positions[i]
                d_ris_rx = np.linalg.norm(ris_pos - rx_pos, axis=2)
                exp_ris_rx = np.exp(-1j * k * d_ris_rx)

                phase_sums[i] = compute_bf_for_rx(exp_ris, exp_tx_ris, exp_ris_rx)

        elif self.model == 'tang_farfield':
            # Compute TX to RIS phase
            ris_rel_pos = ris_pos - ris_center
            tx_pos = tx_pos.reshape(3,)
            vec_tx_ris = (ris_center - tx_pos) / np.linalg.norm(ris_center - tx_pos) # Unitary vector of TX-RIS direction
            dot_product_tx = np.tensordot(ris_rel_pos, vec_tx_ris, axes=([2], [0])) # (Nh, Nv)

            d_tx_ris_center = np.linalg.norm(ris_center - tx_pos)
            exp_tx_ris_center = np.exp(-1j * k * d_tx_ris_center) # (Nh, Nv)
            exp_tx_ris = np.exp(-1j * k * dot_product_tx) * exp_tx_ris_center # (Nh, Nv)

            for i in range(Nrx):
                rx_pos = rx_positions[i]
                rx_pos = rx_pos.reshape(3,)
                vec_ris_rx = (rx_pos - ris_center) / np.linalg.norm(rx_pos - ris_center) # Unitary vector of RIS-RX direction
                dot_product_rx = np.tensordot(ris_rel_pos, vec_ris_rx, axes=([2], [0])) # (Nh, Nv)

                d_ris_center_rx = np.linalg.norm(rx_pos - ris_center)
                exp_ris_center_rx = np.exp(-1j * k * d_ris_center_rx) # (Nh, Nv)
                exp_ris_rx = np.exp(1j * k * dot_product_rx) * exp_ris_center_rx # (Nh, Nv)

                phase_sums[i] = compute_bf_for_rx(exp_ris, exp_tx_ris, exp_ris_rx)

        else:
            raise ValueError("Invalid model. Supported models: 'Tang_general', 'Tang_farfield'.")
        
        return np.abs(phase_sums)**2
    
    def path_gain_multrx(self, rx_positions):
        """
        Compute total path gain for multiple Rx positions due to RIS reflection.

        Parameters:
        -----------
        rx_positions : array-like of shape (Nrx, 3)
            Positions of all receivers.

        Returns:
        --------
        path_gains : ndarray of shape (Nrx,)
            Path gain for each receiver.
        """

        # Ensure rx_positions is at least 2D
        rx_positions = np.atleast_2d(rx_positions)
        Nrx = rx_positions.shape[0]

        # Retrieve RIS and antenna information
        ris_pos = self.ris.positions.copy()  # shape: (Nrow, Ncol, 3)
        ris_center = self.ris.center.copy()  # shape: (3,)
        tx_pos = self.tx_antenna.position.copy()  # shape: (1, 3)

        # Wavenumber
        k = 2 * pi * self.ris.fc / c

        # Antenna gains in linear scale
        g_tx = 10 ** (self.tx_antenna.gain_dbi / 10)
        g_rx = 10 ** (self.rx_antenna.gain_dbi / 10)

        # RIS element size
        dx = self.ris.dx
        dy = self.ris.dy

        # Path gain constant
        gain_constant = g_tx * g_rx * (dx * dy) ** 2 / (16 * pi ** 2)

        # RIS reflection coefficient (Nrow x Ncol complex matrix)
        exp_ris = self.ris.get_all_refl_coef()

        # Avoid float rounding of magnitude greater than 1
        exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)

        # Placeholder for path gains per receiver
        path_gains = np.zeros(Nrx)

        if self.model == 'tang_general':
            # Distance and phase from Tx to RIS
            d_tx_ris = np.linalg.norm(ris_pos - tx_pos, axis=2)
            exp_tx_ris = np.exp(-1j * k * d_tx_ris)

            # Avoid float rounding of magnitude greater than 1
            exp_tx_ris = filt_round_decimals(exp_tx_ris, 1, atol=1e-12, rtol=0)

            for i in range(Nrx):
                rx_pos = rx_positions[i]

                # Distance and phase from RIS to current Rx
                d_ris_rx = np.linalg.norm(ris_pos - rx_pos, axis=2)  # (Nrow, Ncol)
                exp_ris_rx = np.exp(-1j * k * d_ris_rx)

                # Avoid float rounding of magnitude greater than 1
                exp_ris_rx = filt_round_decimals(exp_ris_rx, 1, atol=1e-12, rtol=0)

                # Total complex phase response per element
                exp_total = exp_tx_ris * exp_ris * exp_ris_rx

                # Avoid float rounding of magnitude greater than 1
                exp_total = filt_round_decimals(exp_total, 1, atol=1e-12, rtol=0)

                # Radiation patterns for RIS from Tx and Rx positions
                ris_rad_pat_tx = self.ris.get_rad_pat(tx_pos)        # (Nrow, Ncol)
                ris_rad_pat_rx = self.ris.get_rad_pat(rx_pos)        # (Nrow, Ncol)

                # Flatten RIS positions to shape (Nelem, 3)
                ris_pos_flat = self.ris.positions.reshape(-1, 3)

                # TX radiation pattern toward RIS positions
                tx_rad_pat_flat = self.tx_antenna.get_rad_pat(ris_pos_flat)  # (Nelem,)
                tx_rad_pat = tx_rad_pat_flat.reshape(ris_pos.shape[:2])      # (Nrow, Ncol)

                # RX radiation pattern: simulate a virtual RX antenna pointing at RIS center
                pointing_vec_rx = self.ris.center - rx_pos
                norm_rx = np.linalg.norm(pointing_vec_rx)
                if norm_rx == 0:
                    raise ValueError("Receiver is at RIS center, undefined pointing direction.")
                pointing_unit_rx = pointing_vec_rx / norm_rx

                # Directions from RX to each RIS element
                ris_vectors_rx = ris_pos - rx_pos              # (Nrow, Ncol, 3)
                norms_rx = np.linalg.norm(ris_vectors_rx, axis=2, keepdims=True)
                unit_vectors_rx = ris_vectors_rx / norms_rx    # (Nrow, Ncol, 3)

                # Angle between RX pointing vector and RIS element directions
                cos_ang_rx = np.sum(unit_vectors_rx * pointing_unit_rx, axis=2)
                cos_ang_rx = np.clip(cos_ang_rx, -1.0, 1.0)
                ang_rel_rx = np.arccos(cos_ang_rx)             # (Nrow, Ncol)

                # Compute RX radiation pattern based on cosine model
                rx_rad_pat = np.zeros_like(ang_rel_rx)
                valid = (ang_rel_rx >= 0) & (ang_rel_rx <= np.pi / 2)
                gain_linear_rx = 10 ** (self.rx_antenna.gain_dbi / 10)
                rx_rad_pat[valid] = np.cos(ang_rel_rx[valid]) ** (gain_linear_rx / 2 - 1)

                # Combine all patterns and phase terms
                f_combine = ris_rad_pat_tx * ris_rad_pat_rx * tx_rad_pat * rx_rad_pat

                # Path gain computation
                path_gains[i] = gain_constant * np.abs(
                    np.sum(exp_total * np.sqrt(f_combine) / (d_tx_ris * d_ris_rx))
                ) ** 2

        elif self.model == 'tang_farfield':
            # TX - RIS center distance
            tx_pos = tx_pos.reshape(3,)
            d_tx_ris_center = np.linalg.norm(ris_center - tx_pos)

            # Obtain RIS rad pat for TX respect to RIS center
            dir_vec_tx = (tx_pos - ris_center) / np.linalg.norm(tx_pos - ris_center)  # shape: (3, )
            cos_ang = np.sum(dir_vec_tx * self.ris.normal)
            cos_ang = np.clip(cos_ang, -1.0, 1.0)  # Ensure valid range
            ang_rel = np.arccos(cos_ang)

            # Compute radiation pattern
            rad_pat_tx = 0
            if (ang_rel >= 0) & (ang_rel <= np.pi / 2): # valid angle
                rad_pat_tx = np.cos(ang_rel) ** self.ris.radiation_factor
            
            gain_constant = gain_constant * rad_pat_tx / (d_tx_ris_center ** 2) # TX common factors added

            # Compute TX to RIS phase
            ris_rel_pos = ris_pos - ris_center
            tx_pos = tx_pos.reshape(3,)
            vec_tx_ris = (ris_center - tx_pos) / np.linalg.norm(ris_center - tx_pos) # Unitary vector of TX-RIS direction
            dot_product_tx = np.tensordot(ris_rel_pos, vec_tx_ris, axes=([2], [0])) # (Nh, Nv)

            d_tx_ris_center = np.linalg.norm(ris_center - tx_pos)
            exp_tx_ris_center = np.exp(-1j * k * d_tx_ris_center) # (Nh, Nv)
            exp_tx_ris = np.exp(-1j * k * dot_product_tx) * exp_tx_ris_center # (Nh, Nv)

            # Avoid float rounding of magnitude greater than 1
            exp_tx_ris = filt_round_decimals(exp_tx_ris, 1, atol=1e-12, rtol=0)

            for i in range(Nrx):
                rx_pos = rx_positions[i]
                rx_pos = rx_pos.reshape(3,)

                # RIS center - RX distance
                d_ris_center_rx = np.linalg.norm(rx_pos - ris_center)

                # Obtain RIS rad pat for TX respect to RIS center
                dir_vec_rx = (rx_pos - ris_center) / np.linalg.norm(rx_pos - ris_center)  # shape: (3, )
                cos_ang = np.sum(dir_vec_rx * self.ris.normal)
                cos_ang = np.clip(cos_ang, -1.0, 1.0)  # Ensure valid range
                ang_rel = np.arccos(cos_ang)

                # Compute radiation pattern
                rad_pat_rx = 0
                if (ang_rel >= 0) & (ang_rel <= np.pi / 2): # valid angle
                    rad_pat_rx = np.cos(ang_rel) ** self.ris.radiation_factor
                
                gain_constant_rx = gain_constant * rad_pat_rx / (d_ris_center_rx ** 2) # RX common factors added

                # Compute RIS - RX phase
                vec_ris_rx = (rx_pos - ris_center) / np.linalg.norm(rx_pos - ris_center) # Unitary vector of RIS-RX direction
                dot_product_rx = np.tensordot(ris_rel_pos, vec_ris_rx, axes=([2], [0])) # (Nh, Nv)

                exp_ris_center_rx = np.exp(-1j * k * d_ris_center_rx) # (Nh, Nv)
                exp_ris_rx = np.exp(1j * k * dot_product_rx) * exp_ris_center_rx # (Nh, Nv)

                # Avoid float rounding of magnitude greater than 1
                exp_ris_rx = filt_round_decimals(exp_ris_rx, 1, atol=1e-12, rtol=0)

                exp_total = exp_tx_ris * exp_ris * exp_ris_rx

                # Avoid float rounding of magnitude greater than 1
                exp_total = filt_round_decimals(exp_total, 1, atol=1e-12, rtol=0)

                # Path gain computation
                path_gains[i] = gain_constant_rx * np.abs(np.sum(exp_total)) ** 2

        return path_gains
    
    def plot_phase_bf_2d_map(self, ax1lim, ax2lim, Nax1, Nax2, plane='xz'):
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

        bf = self.phase_bf_multrx(rx_points)
        max_bf = (self.ris.Nrow * self.ris.Ncol)**2

        plt.figure(figsize=(6, 4))
        pcm = plt.pcolormesh(A1, A2, bf.reshape(Nax1, Nax2) / max_bf, shading='auto', cmap='viridis', vmin=0, vmax=1)
        plt.xlabel(f"{plane[0].upper()} (m)")
        plt.ylabel(f"{plane[1].upper()} (m)")
        plt.title(f"Phase beamforming distribution in {plane.upper()} plane, mode '{self.ris.mode}', opt '{self.last_optimizer}'")
        plt.colorbar(pcm, label='Normalized Beamforming')
        plt.xlim(ax1lim)
        plt.ylim(ax2lim)

        ax = plt.gca()
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        plt.grid(True)

        tx = self.tx_antenna.position[0]
        rx = self.rx_antenna.position[0]
        ris_start = self.ris.positions[0, 0]
        ris_end = self.ris.positions[-1, -1]

        plt.plot(tx[i1], tx[i2], 'ro', markersize=8)
        plt.text(tx[i1] + 0.3, tx[i2] + 0.3, 'TX', color='red', fontsize=12, ha='left', va='top')

        plt.plot(rx[i1], rx[i2], 'ro', markersize=8)
        plt.text(rx[i1] + 0.3, rx[i2] + 0.3, 'RX', color='red', fontsize=12, ha='left', va='top')

        plt.plot([ris_start[i1], ris_end[i1]], [ris_start[i2], ris_end[i2]], 'r-', linewidth=8)
        plt.text(ris_start[i1] + 0.5, ris_start[i2] + 0.5, 'RIS', color='red', fontsize=12, ha='left', va='top')
        plt.tight_layout()
        plt.show()
    
    def plot_path_gain_2d_map(self, ax1lim, ax2lim, Nax1, Nax2, plane='xz', vmin=-160, vmax=-80):
        """
        Plots a 2D heatmap of the path gain pattern over a specified plane.

        This function computes and visualizes the spatial distribution of the path gain
        generated by the RIS across a 2D plane (e.g., XZ, YZ, or XY). It also overlays 
        the positions of the transmitter (TX), receiver (RX), and RIS panel on the plot 
        for context.

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

        vmin : int or float, optional (default=-160)
            Value of the lower limit of the pcolormesh plot

        vmax : int or float, optional (default=-80)
            Value of the upper limit of the pcolormesh plot

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

        pg = self.path_gain_multrx(rx_points)

        plt.figure(figsize=(6, 4))
        pcm = plt.pcolormesh(A1, A2, 10 * np.log10(pg.reshape(Nax1, Nax2)), shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.xlabel(f"{plane[0].upper()} (m)")
        plt.ylabel(f"{plane[1].upper()} (m)")
        plt.title(f"Path gain distribution in {plane.upper()} plane, mode '{self.ris.mode}', opt '{self.last_optimizer}'")
        plt.colorbar(pcm, label='Path Gain (dB)')
        plt.xlim(ax1lim)
        plt.ylim(ax2lim)

        ax = plt.gca()
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        plt.grid(True)

        tx = self.tx_antenna.position[0]
        rx = self.rx_antenna.position[0]
        ris_start = self.ris.positions[0, 0]
        ris_end = self.ris.positions[-1, -1]

        plt.plot(tx[i1], tx[i2], 'ro', markersize=8)
        plt.text(tx[i1] + 0.3, tx[i2] + 0.3, 'TX', color='red', fontsize=12, ha='left', va='top')

        plt.plot(rx[i1], rx[i2], 'ro', markersize=8)
        plt.text(rx[i1] + 0.3, rx[i2] + 0.3, 'RX', color='red', fontsize=12, ha='left', va='top')

        plt.plot([ris_start[i1], ris_end[i1]], [ris_start[i2], ris_end[i2]], 'r-', linewidth=8)
        plt.text(ris_start[i1] + 0.5, ris_start[i2] + 0.5, 'RIS', color='red', fontsize=12, ha='left', va='top')
        plt.tight_layout()
        plt.show()

    def plot_phase_bf_angle(self, r, Nr, axis='x'):
        """
        Plots the normalized phase beamforming gain as a function of angle in a 2D plane.

        This function evaluates the beamforming response of the RIS over a circular arc
        centered around the RIS, varying the angle in the specified plane (e.g., XZ, YZ, or XY).
        The plot shows how the beamforming performance varies with respect to the angular direction 
        at a fixed radius `r` from the RIS center.

        Parameters
        ----------
        r : float
            Radius of the arc (distance from the RIS center to the sampling points).
        
        Nr : int
            Number of angular samples between -90° and 90° (i.e., from -π/2 to π/2 radians).
        
        axis : str, optional (default='x')
            Axis that defines the angular plane in combination with the RIS normal.
            For example, if axis='x' and the RIS normal is along 'z', the plot is generated
            in the XZ plane. Must be one of 'x', 'y', or 'z'.

        Returns
        -------
        None
            The function displays the plot but does not return any value.
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax_idx = axis_map[axis]
        normal_idx = np.argmax(np.abs(self.ris.normal))
        perp_idx = list({0, 1, 2} - {ax_idx, normal_idx})[0]

        center = self.ris.center.copy()
        angles = np.linspace(-pi / 2, pi / 2, Nr)

        rx_points = np.zeros((Nr, 3))
        rx_points[:, ax_idx] = center[ax_idx] + r * np.sin(angles)
        rx_points[:, normal_idx] = center[normal_idx] + r * np.cos(angles)
        rx_points[:, perp_idx] = center[perp_idx]  # keep this axis constant

        bf = self.phase_bf_multrx(rx_points)
        max_bf = (self.ris.Nrow * self.ris.Ncol)**2

        plt.figure(figsize=(6, 4))
        plt.plot(np.degrees(angles), bf / max_bf, lw=2)
        plt.xlabel(f"Angle (deg) in {axis.upper()}{['x','y','z'][normal_idx].upper()} plane")
        plt.ylabel("Normalized Beamforming")
        plt.title(f"Phase beamforming vs Angle, mode '{self.ris.mode}', opt '{self.last_optimizer}'")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_path_gain_angle(self, r, Nr, axis='x', vmin=-160, vmax=-80):
        """
        Plots the path gain (in dB) as a function of angle in a 2D plane around the RIS.

        This function evaluates the received path gain over a circular arc of radius `r`
        centered at the RIS, varying the observation angle from −90° to 90° in a specified
        angular plane. The computed path gain includes the full channel response (e.g., with or without RIS effect)
        at each sampling point. This visualization helps characterize the directional performance
        of the RIS or antenna system.

        Parameters
        ----------
        r : float
            Radius of the arc (distance from the RIS center to the sampling points).

        Nr : int
            Number of angular samples between −90° and 90° (in total).

        axis : str, optional (default='x')
            Axis defining the angular scanning plane in combination with the RIS normal vector.
            For example, if `axis='x'` and the RIS normal is along `z`, the scan is performed
            in the XZ plane. Must be one of 'x', 'y', or 'z'.

        vmin : float, optional (default=-160)
            Minimum value (in dB) for the y-axis (used to clip the dynamic range of the plot).

        vmax : float, optional (default=-80)
            Maximum value (in dB) for the y-axis.

        Returns
        -------
        None
            The function displays the plot but does not return any value.
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax_idx = axis_map[axis]
        normal_idx = np.argmax(np.abs(self.ris.normal))
        perp_idx = list({0, 1, 2} - {ax_idx, normal_idx})[0]

        center = self.ris.center.copy()
        angles = np.linspace(-pi / 2, pi / 2, Nr)

        rx_points = np.zeros((Nr, 3))
        rx_points[:, ax_idx] = center[ax_idx] + r * np.sin(angles)
        rx_points[:, normal_idx] = center[normal_idx] + r * np.cos(angles)
        rx_points[:, perp_idx] = center[perp_idx]  # keep this axis constant

        pg = self.path_gain_multrx(rx_points)

        plt.figure(figsize=(6, 4))
        plt.plot(np.degrees(angles), 10 * np.log10(pg), lw=2)
        plt.xlabel(f"Angle (deg) in {axis.upper()}{['x','y','z'][normal_idx].upper()} plane")
        plt.ylabel("Path Gain (dB)")
        plt.title(f"Path Gain vs Angle, mode '{self.ris.mode}', opt '{self.last_optimizer}'")
        plt.ylim([vmin, vmax])
        plt.xlim([-90 , 90])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_phase_bf_dist(self, phi_deg, theta_deg, r1, r2, Nr):
        """
        Plots the normalized phase beamforming gain as a function of distance 
        in a fixed 3D direction from the RIS center.

        This function evaluates the beamforming response of the RIS along a 
        ray starting at the RIS center and pointing in the direction specified 
        by spherical angles (phi, theta). The direction is fixed, and distances 
        are swept from r1 to r2.

        Parameters
        ----------
        phi_deg : float
            Azimuth angle in degrees (measured in the XY plane from +X toward +Y).
            Can be in the range [-180, 180] or [0, 360].

        theta_deg : float
            Elevation angle in degrees (measured from the +Z axis downward).
            Must be in the range [0, 180].

        r1 : float
            Initial distance from the RIS center to start sampling (must be > 0).

        r2 : float
            Final distance from the RIS center to end sampling.

        Nr : int
            Number of distance samples between r1 and r2.

        Returns
        -------
        None
            The function displays the plot but does not return any value.
        """
        # Check theta range
        if not (0 <= theta_deg <= 180):
            raise ValueError("theta_deg must be between 0 and 180 degrees (measured from +Z axis).")

        # Convert angles from degrees to radians
        phi = np.deg2rad(phi_deg % 360)     # Normalize phi to [0, 360)
        theta = np.deg2rad(theta_deg)

        # Direction unit vector in spherical coordinates
        dir_vector = np.array([
            np.sin(theta) * np.cos(phi),  # x
            np.sin(theta) * np.sin(phi),  # y
            np.cos(theta)                 # z
        ])

        # Sampling distances
        r_values = np.linspace(r1, r2, Nr)

        # Generate Rx points along the specified direction
        center = self.ris.center.copy()
        rx_points = center[None, :] + r_values[:, None] * dir_vector[None, :]

        # Compute beamforming gain
        bf = self.phase_bf_multrx(rx_points)
        max_bf = (self.ris.Nrow * self.ris.Ncol) ** 2

        # Plot results
        plt.figure(figsize=(6, 4))
        plt.plot(r_values, bf / max_bf, lw=2)
        plt.xlabel("Distance from RIS center (m)")
        plt.ylabel("Normalized Beamforming")
        plt.title(fr"Phase BF vs Distance at ($\theta=${theta_deg}$^\circ$, $\varphi=${phi_deg}$^\circ$), mode '{self.ris.mode}', opt '{self.last_optimizer}'")
        plt.ylim([0, 1.01])
        plt.xlim([r1, r2])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_path_gain_dist(self, phi_deg, theta_deg, r1, r2, Nr, vmin=-160, vmax=-80):
        """
        Plots the path gain (in dB) as a function of distance from the RIS in a fixed direction.

        This function evaluates the path gain at positions along a line that starts from the RIS center
        and extends in the specified angular direction (phi, theta) for distances ranging from `r1` to `r2`.
        It is useful to characterize how the gain decays with distance in a specific direction.

        Parameters
        ----------
        phi_deg : float
            Azimuth angle in degrees. Measured from X axis in XY plane, counterclockwise.

        theta_deg : float
            Elevation angle in degrees. Measured from Z axis downward to XY plane.

        r1 : float
            Starting distance from the RIS center.

        r2 : float
            Ending distance from the RIS center.

        Nr : int
            Number of samples between r1 and r2.

        vmin : float, optional (default=-160)
            Minimum value (in dB) for the y-axis (used to clip the dynamic range of the plot).

        vmax : float, optional (default=-80)
            Maximum value (in dB) for the y-axis.

        Returns
        -------
        None
            The function displays the plot but does not return any value.
        """
        # Check theta range
        if not (0 <= theta_deg <= 180):
            raise ValueError("theta_deg must be between 0 and 180 degrees (measured from +Z axis).")

        # Convert angles from degrees to radians
        phi = np.deg2rad(phi_deg % 360)     # Normalize phi to [0, 360)
        theta = np.deg2rad(theta_deg)

        # Direction unit vector in spherical coordinates
        dir_vector = np.array([
            np.sin(theta) * np.cos(phi),  # x
            np.sin(theta) * np.sin(phi),  # y
            np.cos(theta)                 # z
        ])

        # Sampling distances
        r_values = np.linspace(r1, r2, Nr)

        # Generate Rx points along the specified direction
        center = self.ris.center.copy()
        rx_points = center[None, :] + r_values[:, None] * dir_vector[None, :]

        # Evaluate path gain
        pg = self.path_gain_multrx(rx_points)  # Expected shape: (Nr,)

        # Plot results
        plt.figure(figsize=(6, 4))
        plt.plot(r_values, 10 * np.log10(pg), lw=2)
        plt.xlabel("Distance from RIS center (m)")
        plt.ylabel("Path Gain (dB)")
        plt.title(fr"Path Gain vs Distance at ($\theta=${theta_deg}$^\circ$, $\varphi=${phi_deg}$^\circ$), mode '{self.ris.mode}', opt '{self.last_optimizer}'")
        plt.ylim([vmin, vmax])
        plt.xlim([r1, r2])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_phase_bf_angle_2d_map(self, r, Nr):
        """
        Plots a 2D angular map of normalized phase beamforming in spherical coordinates.

        This function computes and visualizes the beamforming gain over a spherical surface 
        centered at the RIS, at a fixed radius `r`. The gain is sampled as a function of 
        azimuth (φ) and elevation (θ) angles, based on the boresight direction of the RIS 
        ('x', 'y', or 'z'). The resulting heatmap shows how beamforming varies with direction 
        in angular space.

        Parameters
        ----------
        r : float
            Radius of the observation sphere (distance from the RIS center to sampling points).

        Nr : int
            Number of angular samples for both azimuth and elevation (resolution of the map).

        Returns
        -------
        None
            The function displays a 2D color map of normalized beamforming gain
            as a function of azimuth and elevation angles. No value is returned.
        """
        ris_center = self.ris.center.copy()
        boresight = self.ris.boresight.lower()

        if boresight == 'z':
            theta = np.linspace(0, pi / 2, Nr)  # elevation
            phi = np.linspace(-pi, pi, Nr)   # azimuth
        elif boresight == 'x':
            theta = np.linspace(0, pi, Nr)  # elevation
            phi = np.linspace(-pi / 2, pi / 2, Nr)   # azimuth
        else:   # boresight == 'y'
            theta = np.linspace(0, pi, Nr)  # elevation
            phi = np.linspace(0, pi, Nr)   # azimuth

        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

        x = r * np.sin(Theta) * np.cos(Phi)
        y = r * np.sin(Theta) * np.sin(Phi)
        z = r * np.cos(Theta)

        rx_points = np.stack([x, y, z], axis=-1) + ris_center
        rx_points = rx_points.reshape(-1, 3)

        bf = self.phase_bf_multrx(rx_points)
        max_bf = (self.ris.Nrow * self.ris.Ncol)**2

        plt.figure(figsize=(6, 4))
        pcm = plt.pcolormesh(np.degrees(Phi), np.degrees(Theta), bf.reshape(Nr, Nr) / max_bf, shading='auto', cmap='viridis', vmin=0, vmax=1)
        plt.xlabel(r'Azimuth $\varphi$ (deg)')
        plt.ylabel(r'Elevation $\theta$ (deg)')
        plt.title(f"2D Angular Phase Beamforming Map, mode '{self.ris.mode}', opt '{self.last_optimizer}'")
        plt.colorbar(pcm, label='Normalized Beamforming')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_path_gain_angle_2d_map(self, r, Nr, vmin=-160, vmax=-80):
        """
        Plots a 2D angular map of the path gain (in dB) in spherical coordinates around the RIS.

        This function evaluates and visualizes the path gain as a function of azimuth (φ) and 
        elevation (θ) angles over a spherical surface of radius `r` centered at the RIS. 
        The angular scanning range and direction are determined by the RIS boresight axis 
        ('x', 'y', or 'z'). The resulting heatmap provides an angular characterization 
        of the channel strength, which may include direct and/or RIS-reflected paths.

        Parameters
        ----------
        r : float
            Radius of the observation sphere (distance from the RIS center to the evaluation points).

        Nr : int
            Number of angular samples used in both azimuth and elevation directions 
            (controls the resolution of the angular map).

        vmin : float, optional (default=-160)
            Minimum value (in dB) for the color scale of the path gain plot.

        vmax : float, optional (default=-80)
            Maximum value (in dB) for the color scale of the path gain plot.

        Returns
        -------
        None
            The function displays a 2D heatmap of path gain in dB versus azimuth and elevation.
            It does not return any value.
        """
        ris_center = self.ris.center.copy()
        boresight = self.ris.boresight.lower()

        if boresight == 'z':
            theta = np.linspace(0, pi / 2, Nr)  # elevation
            phi = np.linspace(-pi, pi, Nr)   # azimuth
        elif boresight == 'x':
            theta = np.linspace(0, pi, Nr)  # elevation
            phi = np.linspace(-pi / 2, pi / 2, Nr)   # azimuth
        else:   # boresight == 'y'
            theta = np.linspace(0, pi, Nr)  # elevation
            phi = np.linspace(0, pi, Nr)   # azimuth

        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

        x = r * np.sin(Theta) * np.cos(Phi)
        y = r * np.sin(Theta) * np.sin(Phi)
        z = r * np.cos(Theta)

        rx_points = np.stack([x, y, z], axis=-1) + ris_center
        rx_points = rx_points.reshape(-1, 3)

        path_gains = self.path_gain_multrx(rx_points)
        path_gains_db = 10 * np.log10(path_gains.reshape(Nr, Nr))

        plt.figure(figsize=(6, 4))
        pcm = plt.pcolormesh(np.degrees(Phi), np.degrees(Theta), path_gains_db, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.xlabel(r'Azimuth $\varphi$ (deg)')
        plt.ylabel(r'Elevation $\theta$ (deg)')
        plt.title(f"2D Angular Path Gain Map, mode '{self.ris.mode}', opt '{self.last_optimizer}'")
        plt.colorbar(pcm, label='Path Gain (dB)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_phase_bf_angle_3d(self, r, Nr):
        """
        Plots a 3D representation of the normalized phase beamforming pattern in spherical coordinates.

        This function computes the RIS beamforming gain over a spherical surface of radius `r`
        centered at the RIS. The beamforming response is evaluated as a function of azimuth (φ)
        and elevation (θ) angles, with angular coverage determined by the RIS boresight direction
        ('x', 'y', or 'z'). The resulting pattern is plotted in 3D using Plotly, where the distance
        from the origin in each direction is proportional to the beamforming gain in that direction.
        The RIS plane is also visualized for spatial reference.

        Parameters
        ----------
        r : float
            Radius of the observation sphere (distance from the RIS center to the sampling surface).

        Nr : int
            Number of samples used in both azimuth and elevation angles (defines angular resolution).

        Returns
        -------
        None
            The function renders an interactive 3D surface plot of the normalized beamforming pattern.
            No values are returned.
        """
        ris_center = self.ris.center.copy()
        boresight = self.ris.boresight.lower()

        # Angles depending on boresight direction
        if boresight == 'z':
            theta = np.linspace(0, pi / 2, Nr)
            phi = np.linspace(-pi, pi, Nr)
        elif boresight == 'x':
            theta = np.linspace(0, pi, Nr)
            phi = np.linspace(-pi / 2, pi / 2, Nr)
        else:  # boresight == 'y'
            theta = np.linspace(0, pi, Nr)
            phi = np.linspace(0, pi, Nr)

        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

        # Cartesian coords of the receivers
        x = r * np.sin(Theta) * np.cos(Phi)
        y = r * np.sin(Theta) * np.sin(Phi)
        z = r * np.cos(Theta)

        rx_points = np.stack([x, y, z], axis=-1) + ris_center
        rx_points = rx_points.reshape(-1, 3)

        # Normalized Beamforming
        bf = self.phase_bf_multrx(rx_points)
        max_bf = (self.ris.Nrow * self.ris.Ncol) ** 2
        bf_norm = bf / max_bf + 1e-12  # Evitar ceros para log o división

        # To shape (Nr, Nr)
        BF = bf_norm.reshape(Nr, Nr)

        # Unitary direction of each point
        vx = np.sin(Theta) * np.cos(Phi)
        vy = np.sin(Theta) * np.sin(Phi)
        vz = np.cos(Theta)

        # Cartesiand cords scaled with the beamforming
        X = BF * vx
        Y = BF * vy
        Z = BF * vz

        # Extract two base vectors from the RIS plane
        positions = self.ris.positions  # shape (Nrow, Ncol, 3)
        v1 = positions[0, -1] - positions[0, 0]  # horizontal axis
        v2 = positions[-1, 0] - positions[0, 0]  # vertical axis

        # Create mesh for the plane
        s = np.linspace(-0.5, 0.5, 2)
        t = np.linspace(-0.5, 0.5, 2)
        S, T = np.meshgrid(s, t)
        plane = self.ris.center.reshape(1, 1, 3) + S[..., None] * v1 + T[..., None] * v2

        Xp = plane[..., 0]
        Yp = plane[..., 1]
        Zp = plane[..., 2]

        # Plot with Plotly
        fig = go.Figure(data=[
            go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=BF,
                colorbar=dict(
                    title=dict(
                        text="Normalized Beamforming",
                        side='right',    # vertical title
                        font=dict(size=14)
                    )
                ),
                colorscale='Viridis',
                cmin=0,
                cmax=1,
                showscale=True,
                opacity=0.9
            )
        ])

        fig.update_layout(
            title='3D RIS Phase Beamforming Pattern',
            width=600,
            height=500,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                zaxis=dict(range=[-1, 1]),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.35)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=25),
        )

        # Add the plane as an opaque red plane
        fig.add_trace(go.Surface(
            x=Xp, y=Yp, z=Zp,
            showscale=False,
            opacity=0.8,
            surfacecolor=np.ones_like(Xp),  # plane color
            colorscale=[[0, 'red'], [1, 'red']],
            name='RIS Plane',
            hoverinfo='skip'
        ))

        fig.show()

    def __repr__(self):
        return (f"tx_ris_rx_channel(model='{self.model}', "
                f"RIS size={self.ris.Nrow}x{self.ris.Ncol})")
