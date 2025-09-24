import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi

from surpRISe.utils.utils import *
from surpRISe.optimizers.tx_ris_rx_opt import tx_ris_rx_direct_disc_opt, tx_ris_rx_max_bf_opt
from surpRISe.components.antenna import Antenna
from surpRISe.components.ris import RIS

import plotly.io as pio
pio.renderers.default = 'notebook_connected' # 'browser' for avoid error but shown in the browser
import plotly.graph_objects as go

import scienceplots
plt.style.use(['science','no-latex'])
plt.rcParams.update({'font.size': 15})

class tang_ff_channel:
    def __init__(self):
        """
        Initialize Tang Farfield Channel Model
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

            # Avoid float rounding of magnitude greater than 1
            exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)
            exp_tx_ris = filt_round_decimals(exp_tx_ris, 1, atol=1e-12, rtol=0)
            exp_ris_rx = filt_round_decimals(exp_ris_rx, 1, atol=1e-12, rtol=0)

            # Element-wise product and sum
            exp_total = exp_tx_ris * exp_ris * exp_ris_rx

            # Avoid float rounding of magnitude greater than 1
            exp_total = filt_round_decimals(exp_total, 1, atol=1e-12, rtol=0)

            phase_sums[i] = np.sum(exp_total)


        return np.abs(phase_sums)**2
    
    def path_gain_multrx(self, tx_antenna=None, rx_antenna=None, rx_positions=None, ris=None):
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
        rx_positions : ndarray of shape (Nrx, 3)
            Array with the 3D positions of the receivers.
        ris : object
            RIS object containing geometry (positions, element spacing, center),
            carrier frequency, reflection coefficients, and radiation patterns.

        Returns
        -------
        path_gains : ndarray of shape (Nrx,)
            Path gain in linear scale at each receiver location.
        """

        if tx_antenna is None or rx_antenna is None or rx_positions is None or ris is None:
            raise ValueError("tx_antenna, rx_antenna, rx_positions, and ris must be provided.")

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
        gain_constant = g_tx * g_rx * (dx * dy) ** 2 / (16 * pi ** 2)

        # RIS reflection coefficient (Nrow x Ncol complex matrix)
        exp_ris = ris.get_all_refl_coef()

        # Avoid float rounding of magnitude greater than 1
        exp_ris = filt_round_decimals(exp_ris, 1, atol=1e-12, rtol=0)

        # Placeholder for path gains per receiver
        path_gains = np.zeros(Nrx)

        # TX - RIS center distance
        tx_pos = tx_pos.reshape(3,)
        d_tx_ris_center = np.linalg.norm(ris_center - tx_pos)

        # Obtain RIS rad pat for TX respect to RIS center
        dir_vec_tx = (tx_pos - ris_center) / np.linalg.norm(tx_pos - ris_center)  # shape: (3, )
        cos_ang = np.sum(dir_vec_tx * ris.normal)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)  # Ensure valid range
        ang_rel = np.arccos(cos_ang)

        # Compute radiation pattern
        rad_pat_tx = 0
        if (ang_rel >= 0) & (ang_rel <= np.pi / 2): # valid angle
            rad_pat_tx = np.cos(ang_rel) ** ris.radiation_factor
        
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
            cos_ang = np.sum(dir_vec_rx * ris.normal)
            cos_ang = np.clip(cos_ang, -1.0, 1.0)  # Ensure valid range
            ang_rel = np.arccos(cos_ang)

            # Compute radiation pattern
            rad_pat_rx = 0
            if (ang_rel >= 0) & (ang_rel <= np.pi / 2): # valid angle
                rad_pat_rx = np.cos(ang_rel) ** ris.radiation_factor
            
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

    def plot_phase_bf_2d_map(self, tx_antenna=None, ris=None, ax1lim=None, ax2lim=None, Nax1=None, Nax2=None, plane='xz'):
        """
        Plot a 2D heatmap of the normalized beamforming distribution over a specified plane.

        This function computes the phase beamforming response produced by the RIS
        for a grid of virtual receiver (RX) positions across a given 2D plane 
        (e.g., XZ, YZ, or XY). The results are normalized with respect to the 
        maximum theoretical gain and visualized as a color map. The positions of 
        the transmitter (TX) and RIS are overlaid for reference.

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object containing its position attribute.
        ris : object
            RIS object containing its geometry, mode, and element configuration.
        ax1lim : list or tuple of float, shape (2,)
            Limits of the first axis of the plane [min, max].
        ax2lim : list or tuple of float, shape (2,)
            Limits of the second axis of the plane [min, max].
        Nax1 : int
            Number of sampling points along the first axis.
        Nax2 : int
            Number of sampling points along the second axis.
        plane : str, optional (default='xz')
            Plane over which the map is computed. Must be two characters from 
            'x', 'y', or 'z' (e.g., 'xz', 'yz', 'xy').

        Returns
        -------
        None
            Displays a 2D color map plot of the normalized beamforming distribution.
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

    def plot_path_gain_2d_map(self, tx_antenna=None, rx_antenna=None, ris=None, ax1lim=None, ax2lim=None, Nax1=None, Nax2=None, plane='xz', vmin=-160, vmax=-80):
        """
        Plot a 2D heatmap of the RIS-assisted path gain distribution over a specified plane.

        This function computes the path gain for a dense grid of virtual receiver (RX) 
        positions across a given 2D plane (e.g., XZ, YZ, or XY) using the TX → RIS → RX 
        propagation path. The results, expressed in dB, are visualized as a color map. 
        The positions of the transmitter (TX) and the RIS panel are overlaid for reference.

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object containing position and gain attributes.
        rx_antenna : object
            Receiver antenna object containing gain attributes.
        ris : object
            RIS object containing geometry, mode, reflection coefficients, and element spacing.
        ax1lim : list or tuple of float, shape (2,)
            Limits of the first axis of the plane [min, max].
        ax2lim : list or tuple of float, shape (2,)
            Limits of the second axis of the plane [min, max].
        Nax1 : int
            Number of sampling points along the first axis.
        Nax2 : int
            Number of sampling points along the second axis.
        plane : str, optional (default='xz')
            Plane over which the map is computed. Must be two characters from 
            'x', 'y', or 'z' (e.g., 'xz', 'yz', or 'xy').
        vmin : int or float, optional (default=-160)
            Lower limit (in dB) of the color scale in the plot.
        vmax : int or float, optional (default=-80)
            Upper limit (in dB) of the color scale in the plot.

        Returns
        -------
        None
            Displays a 2D color map plot of the RIS-assisted path gain distribution.
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

        pg = self.path_gain_multrx(tx_antenna=tx_antenna, rx_antenna=rx_antenna, rx_positions=rx_points, ris=ris)

        plt.figure(figsize=(6, 4))
        pcm = plt.pcolormesh(A1, A2, 10 * np.log10(pg.reshape(Nax1, Nax2)), shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.xlabel(f"{plane[0].upper()} (m)")
        plt.ylabel(f"{plane[1].upper()} (m)")
        plt.title(f"Path gain distribution in {plane.upper()} plane, mode '{ris.mode}', opt '{self.last_optimizer}'")
        plt.colorbar(pcm, label='Path Gain (dB)')
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

    def plot_phase_bf_angle(self, ris, tx_antenna, r, Nr, axis='x'):
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
        normal_idx = np.argmax(np.abs(ris.normal))
        perp_idx = list({0, 1, 2} - {ax_idx, normal_idx})[0]

        center = ris.center.copy()
        angles = np.linspace(-pi / 2, pi / 2, Nr)

        rx_points = np.zeros((Nr, 3))
        rx_points[:, ax_idx] = center[ax_idx] + r * np.sin(angles)
        rx_points[:, normal_idx] = center[normal_idx] + r * np.cos(angles)
        rx_points[:, perp_idx] = center[perp_idx]  # keep this axis constant

        bf = self.phase_bf_multrx(tx_antenna=tx_antenna, rx_positions=rx_points, ris=ris)
        max_bf = (ris.Nrow * ris.Ncol)**2

        plt.figure(figsize=(6, 4))
        plt.plot(np.degrees(angles), bf / max_bf, lw=2)
        plt.xlabel(f"Angle (deg) in {axis.upper()}{['x','y','z'][normal_idx].upper()} plane")
        plt.ylabel("Normalized Beamforming")
        plt.title(f"Phase beamforming vs Angle, mode '{ris.mode}', opt '{self.last_optimizer}'")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_path_gain_angle(self, tx_antenna, rx_antenna, ris, r, Nr, axis='x', vmin=-160, vmax=-80):
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
        normal_idx = np.argmax(np.abs(ris.normal))
        perp_idx = list({0, 1, 2} - {ax_idx, normal_idx})[0]

        center = ris.center.copy()
        angles = np.linspace(-pi / 2, pi / 2, Nr)

        rx_points = np.zeros((Nr, 3))
        rx_points[:, ax_idx] = center[ax_idx] + r * np.sin(angles)
        rx_points[:, normal_idx] = center[normal_idx] + r * np.cos(angles)
        rx_points[:, perp_idx] = center[perp_idx]  # keep this axis constant

        pg = self.path_gain_multrx(tx_antenna=tx_antenna, rx_antenna=rx_antenna, rx_positions=rx_points, ris=ris)

        plt.figure(figsize=(6, 4))
        plt.plot(np.degrees(angles), 10 * np.log10(pg), lw=2)
        plt.xlabel(f"Angle (deg) in {axis.upper()}{['x','y','z'][normal_idx].upper()} plane")
        plt.ylabel("Path Gain (dB)")
        plt.title(f"Path Gain vs Angle, mode '{ris.mode}', opt '{self.last_optimizer}'")
        plt.ylim([vmin, vmax])
        plt.xlim([-90 , 90])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_phase_bf_dist(self, tx_antenna, ris, phi_deg, theta_deg, r1, r2, Nr):
        """
        Plots the normalized phase beamforming gain as a function of distance 
        along a fixed 3D direction from the RIS center.

        This function evaluates how the RIS beamforming response evolves with 
        distance when steering the beam toward a direction defined by spherical 
        angles (phi, theta). The response is sampled along a straight line (ray) 
        that originates at the RIS center and extends between distances r1 and r2.

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object providing the source position and parameters.

        ris : object
            RIS object defining the panel geometry, center, and operating mode.

        phi_deg : float
            Azimuth angle in degrees (measured in the XY-plane from +X toward +Y).
            Can be in the range [-180, 180] or [0, 360].

        theta_deg : float
            Elevation angle in degrees (measured from the +Z axis downward).
            Must be in the range [0, 180].

        r1 : float
            Starting distance from the RIS center along the ray (must be > 0).

        r2 : float
            Ending distance from the RIS center along the ray.

        Nr : int
            Number of sample points between r1 and r2.

        Returns
        -------
        None
            Displays a 2D line plot of the normalized phase beamforming gain as 
            a function of distance. No values are returned.
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
        center = ris.center.copy()
        rx_points = center[None, :] + r_values[:, None] * dir_vector[None, :]

        # Compute beamforming gain
        bf = self.phase_bf_multrx(tx_antenna=tx_antenna, rx_positions=rx_points, ris=ris)
        max_bf = (ris.Nrow * ris.Ncol) ** 2

        # Plot results
        plt.figure(figsize=(6, 4))
        plt.plot(r_values, bf / max_bf, lw=2)
        plt.xlabel("Distance from RIS center (m)")
        plt.ylabel("Normalized Beamforming")
        plt.title(fr"Phase BF vs Distance at ($\theta=${theta_deg}$^\circ$, $\varphi=${phi_deg}$^\circ$), mode '{ris.mode}', opt '{self.last_optimizer}'")
        plt.ylim([0, 1.01])
        plt.xlim([r1, r2])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_path_gain_dist(self, tx_antenna, rx_antenna, ris, phi_deg, theta_deg, r1, r2, Nr, vmin=-160, vmax=-80):
        """
        Plots the path gain (in dB) as a function of distance from the RIS center 
        along a fixed 3D direction.

        This function computes the path gain at receiver positions distributed 
        along a straight line (ray) that originates at the RIS center and points 
        toward a direction defined by spherical angles (phi, theta). Distances 
        from `r1` to `r2` are sampled uniformly, and the resulting path gain is 
        plotted in dB scale. This allows analyzing how the channel strength decays 
        with distance in a specific angular direction.

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object providing the source position and parameters.

        rx_antenna : object
            Receiver antenna object used to evaluate the propagation response.

        ris : object
            RIS object defining the geometry, center position, and operating mode.

        phi_deg : float
            Azimuth angle in degrees (measured in the XY plane from +X toward +Y).
            Accepts values in the range [-180, 180] or [0, 360].

        theta_deg : float
            Elevation angle in degrees (measured from the +Z axis downward).
            Must lie within [0, 180].

        r1 : float
            Starting distance from the RIS center along the specified direction.

        r2 : float
            Ending distance from the RIS center along the specified direction.

        Nr : int
            Number of distance samples between r1 and r2.

        vmin : float, optional (default=-160)
            Minimum y-axis limit in dB (controls the dynamic range of the plot).

        vmax : float, optional (default=-80)
            Maximum y-axis limit in dB.

        Returns
        -------
        None
            Displays a 2D line plot of the path gain (in dB) versus distance. 
            The function produces a visualization but does not return any value.
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
        center = ris.center.copy()
        rx_points = center[None, :] + r_values[:, None] * dir_vector[None, :]

        # Evaluate path gain
        pg = self.path_gain_multrx(tx_antenna=tx_antenna, rx_antenna=rx_antenna, rx_positions=rx_points, ris=ris)  # Expected shape: (Nr,)

        # Plot results
        plt.figure(figsize=(6, 4))
        plt.plot(r_values, 10 * np.log10(pg), lw=2)
        plt.xlabel("Distance from RIS center (m)")
        plt.ylabel("Path Gain (dB)")
        plt.title(fr"Path Gain vs Distance at ($\theta=${theta_deg}$^\circ$, $\varphi=${phi_deg}$^\circ$), mode '{ris.mode}', opt '{self.last_optimizer}'")
        plt.ylim([vmin, vmax])
        plt.xlim([r1, r2])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_phase_bf_angle_2d_map(self, tx_antenna, ris, r, Nr):
        """
        Plots a 2D angular heatmap of the normalized phase beamforming gain 
        over a spherical surface centered at the RIS.

        This function evaluates the beamforming response on a sphere of radius `r` 
        centered at the RIS. The gain is sampled as a function of azimuth (φ) and 
        elevation (θ) angles, where the angular ranges depend on the RIS boresight 
        direction ('x', 'y', or 'z'). The resulting 2D map shows how the RIS 
        distributes its beamforming capability across directions in angular space.

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object providing the source position and parameters.

        ris : object
            RIS object defining the geometry, center position, boresight orientation, 
            and operating mode.

        r : float
            Radius of the observation sphere (distance from the RIS center to the 
            sampling points).

        Nr : int
            Number of samples along both azimuth and elevation (controls the angular 
            resolution of the map).

        Returns
        -------
        None
            Displays a 2D color heatmap of normalized beamforming gain as a function 
            of azimuth and elevation angles. The function produces a visualization 
            but does not return any value.
        """
        ris_center = ris.center.copy()
        boresight = ris.boresight.lower()

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

        bf = self.phase_bf_multrx(tx_antenna=tx_antenna, rx_positions=rx_points, ris=ris)
        max_bf = (ris.Nrow * ris.Ncol)**2

        plt.figure(figsize=(6, 4))
        pcm = plt.pcolormesh(np.degrees(Phi), np.degrees(Theta), bf.reshape(Nr, Nr) / max_bf, shading='auto', cmap='viridis', vmin=0, vmax=1)
        plt.xlabel(r'Azimuth $\varphi$ (deg)')
        plt.ylabel(r'Elevation $\theta$ (deg)')
        plt.title(f"2D Angular Phase Beamforming Map, mode '{ris.mode}', opt '{self.last_optimizer}'")
        plt.colorbar(pcm, label='Normalized Beamforming')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_path_gain_angle_2d_map(self, tx_antenna, rx_antenna, ris, r, Nr, vmin=-160, vmax=-80):
        """
        Plots a 2D angular heatmap of the path gain (in dB) in spherical coordinates 
        around the RIS.

        This function evaluates the channel path gain over a spherical surface of 
        radius `r` centered at the RIS. The gain is computed as a function of azimuth 
        (φ) and elevation (θ) angles, where the angular ranges are determined by the 
        RIS boresight direction ('x', 'y', or 'z'). The resulting heatmap shows how 
        the effective channel strength varies with direction, including both direct 
        and RIS-reflected contributions.

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object, providing the position and gain pattern of the TX.

        rx_antenna : object
            Receiver antenna object, defining the position, orientation, and gain 
            pattern of the RX.

        ris : object
            RIS object containing the geometry, element configuration, center position, 
            boresight orientation, and operating mode.

        r : float
            Radius of the observation sphere (distance from the RIS center to the 
            sampling points).

        Nr : int
            Number of samples along both azimuth and elevation directions 
            (sets the angular resolution of the map).

        vmin : float, optional (default=-160)
            Minimum path gain value (in dB) used to clip the color scale of the plot.

        vmax : float, optional (default=-80)
            Maximum path gain value (in dB) used for the color scale.

        Returns
        -------
        None
            Displays a 2D heatmap of path gain (in dB) versus azimuth and elevation 
            angles. The function produces a visualization but does not return any value.
        """
        ris_center = ris.center.copy()
        boresight = ris.boresight.lower()

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

        path_gains = self.path_gain_multrx(tx_antenna=tx_antenna, rx_antenna=rx_antenna, rx_positions=rx_points, ris=ris)
        path_gains_db = 10 * np.log10(path_gains.reshape(Nr, Nr))

        plt.figure(figsize=(6, 4))
        pcm = plt.pcolormesh(np.degrees(Phi), np.degrees(Theta), path_gains_db, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.xlabel(r'Azimuth $\varphi$ (deg)')
        plt.ylabel(r'Elevation $\theta$ (deg)')
        plt.title(f"2D Angular Path Gain Map, mode '{ris.mode}', opt '{self.last_optimizer}'")
        plt.colorbar(pcm, label='Path Gain (dB)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_phase_bf_angle_2d_polar(self, tx_antenna, ris, r, Nr):
        """
        Plots a 2D polar map of the normalized phase beamforming gain 
        over a spherical surface centered at the RIS.

        This function computes the beamforming gain at points on a 
        spherical grid around the RIS, given a transmit antenna and 
        the RIS configuration, and visualizes it in a polar plot.

        Parameters
        ----------
        tx_antenna : object
            The transmit antenna object used to compute the beamforming.
        ris : object
            The RIS object containing parameters like center, boresight,
            number of rows and columns, and mode.
        r : float
            Radius of the spherical surface where the beamforming is evaluated.
        Nr : int
            Number of angular points in both theta (elevation) and phi (azimuth) directions.

        Returns
        -------
        None
            Displays a polar plot with normalized beamforming gain.
        """
        ris_center = ris.center.copy()
        boresight = ris.boresight.lower()

        # Define angle ranges depending on boresight
        if boresight == 'z':
            theta = np.linspace(0, np.pi/2, Nr)   # elevation (0 -> perpendicular to RIS, pi/2 -> horizon)
            phi = np.linspace(-np.pi, np.pi, Nr)  # azimuth
        elif boresight == 'x':
            theta = np.linspace(0, np.pi/2, Nr)
            phi = np.linspace(-np.pi/2, np.pi/2, Nr)
        else:  # boresight == 'y'
            theta = np.linspace(0, np.pi/2, Nr)
            phi = np.linspace(0, np.pi, Nr)

        # Angular mesh
        Phi, Theta = np.meshgrid(phi, theta, indexing="ij")

        # Cartesian coordinates of receiver points
        x = r * np.sin(Theta) * np.cos(Phi)
        y = r * np.sin(Theta) * np.sin(Phi)
        z = r * np.cos(Theta)

        rx_points = np.stack([x, y, z], axis=-1) + ris_center
        rx_points = rx_points.reshape(-1, 3)

        # Compute normalized beamforming
        bf = self.phase_bf_multrx(tx_antenna=tx_antenna, rx_positions=rx_points, ris=ris)
        max_bf = (ris.Nrow * ris.Ncol) ** 2
        bf_norm = bf.reshape(len(phi), len(theta)) / max_bf

        # Polar plot (phi = angle, theta = radius)
        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot(111, polar=True)

        pcm = ax.pcolormesh(Phi, Theta, bf_norm, cmap='viridis', shading='auto', vmin=0, vmax=1)

        # Phi labels (azimuth)
        phi_ticks = np.arange(-np.pi, np.pi, np.pi/6)
        phi_labels = [r'$\varphi=-\pi$', r'$\varphi=-5\pi/6$', r'$\varphi=-2\pi/3$', r'$\varphi=-\pi/2$',
                    r'$\varphi=-\pi/3$', r'$\varphi=-\pi/6$', r'$\varphi=0$', r'$\varphi=\pi/6$',
                    r'$\varphi=\pi/3$', r'$\varphi=\pi/2$', r'$\varphi=2\pi/3$', r'$\varphi=5\pi/6$']
        ax.set_xticks(phi_ticks)
        ax.set_xticklabels(phi_labels)
        ax.tick_params(axis='x', pad=30)  # increase distance of phi labels from plot

        # Theta labels (radius)
        theta_ticks = np.arange(0, np.pi/2+1e-6, np.pi/6)
        theta_labels = [r'$\theta=0$', r'$\theta=\pi/6$', r'$\theta=\pi/3$', r'$\theta=\pi/2$']
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels(theta_labels)

        # Set radial labels color to white
        for label in ax.get_yticklabels():
            label.set_color("white")

        # Adjust limits to not cut the map
        ax.set_thetamin(-180)  # degrees -> -π
        ax.set_thetamax(180)   # degrees -> π
        ax.set_rmax(np.pi/2)   # maximum radius = 90° elevation

        # Title and colorbar
        ax.set_title(f"Polar Phase Beamforming, mode '{ris.mode}', opt '{self.last_optimizer}'", va='bottom', y=1.20)
        cbar = plt.colorbar(pcm, ax=ax, pad=0.2)
        cbar.set_label("Normalized Beamforming")

        plt.show()

    def plot_path_gain_angle_2d_polar(self, tx_antenna, rx_antenna, ris, r, Nr, vmin=-160, vmax=-60):
        """
        Plots a 2D polar map of the path gain over a spherical
        surface centered at the RIS.

        This function computes the path gain at points on a 
        spherical grid around the RIS, given a transmit antenna and 
        the RIS configuration, and visualizes it in a polar plot.

        Parameters
        ----------
        tx_antenna : object
            Transmitter antenna object, providing the position and gain pattern of the TX.

        rx_antenna : object
            Receiver antenna object, defining the position, orientation, and gain 
            pattern of the RX.

        ris : object
            RIS object containing the geometry, element configuration, center position, 
            boresight orientation, and operating mode.

        r : float
            Radius of the observation sphere (distance from the RIS center to the 
            sampling points).

        Nr : int
            Number of samples along both azimuth and elevation directions 
            (sets the angular resolution of the map).

        vmin : float, optional (default=-160)
            Minimum path gain value (in dB) used to clip the color scale of the plot.

        vmax : float, optional (default=-80)
            Maximum path gain value (in dB) used for the color scale.

        Returns
        -------
        None
            Displays a polar plot with normalized beamforming gain.
        """
        ris_center = ris.center.copy()
        boresight = ris.boresight.lower()

        # Define angle ranges depending on boresight
        if boresight == 'z':
            theta = np.linspace(0, np.pi/2, Nr)   # elevation (0 -> perpendicular to RIS, pi/2 -> horizon)
            phi = np.linspace(-np.pi, np.pi, Nr)  # azimuth
        elif boresight == 'x':
            theta = np.linspace(0, np.pi/2, Nr)
            phi = np.linspace(-np.pi/2, np.pi/2, Nr)
        else:  # boresight == 'y'
            theta = np.linspace(0, np.pi/2, Nr)
            phi = np.linspace(0, np.pi, Nr)

        # Angular mesh
        Phi, Theta = np.meshgrid(phi, theta, indexing="ij")

        # Cartesian coordinates of receiver points
        x = r * np.sin(Theta) * np.cos(Phi)
        y = r * np.sin(Theta) * np.sin(Phi)
        z = r * np.cos(Theta)

        rx_points = np.stack([x, y, z], axis=-1) + ris_center
        rx_points = rx_points.reshape(-1, 3)

        # Compute normalized beamforming
        path_gains = self.path_gain_multrx(tx_antenna=tx_antenna, rx_antenna=rx_antenna, rx_positions=rx_points, ris=ris)
        path_gains_db = 10 * np.log10(path_gains.reshape(Nr, Nr))

        # Polar plot (phi = angle, theta = radius)
        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot(111, polar=True)

        pcm = ax.pcolormesh(Phi, Theta, path_gains_db, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)

        # Phi labels (azimuth)
        phi_ticks = np.arange(-np.pi, np.pi, np.pi/6)
        phi_labels = [r'$\varphi=-\pi$', r'$\varphi=-5\pi/6$', r'$\varphi=-2\pi/3$', r'$\varphi=-\pi/2$',
                    r'$\varphi=-\pi/3$', r'$\varphi=-\pi/6$', r'$\varphi=0$', r'$\varphi=\pi/6$',
                    r'$\varphi=\pi/3$', r'$\varphi=\pi/2$', r'$\varphi=2\pi/3$', r'$\varphi=5\pi/6$']
        ax.set_xticks(phi_ticks)
        ax.set_xticklabels(phi_labels)
        ax.tick_params(axis='x', pad=30)  # increase distance of phi labels from plot

        # Theta labels (radius)
        theta_ticks = np.arange(0, np.pi/2+1e-6, np.pi/6)
        theta_labels = [r'$\theta=0$', r'$\theta=\pi/6$', r'$\theta=\pi/3$', r'$\theta=\pi/2$']
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels(theta_labels)

        # Set radial labels color to white
        for label in ax.get_yticklabels():
            label.set_color("white")

        # Adjust limits to not cut the map
        ax.set_thetamin(-180)  # degrees -> -π
        ax.set_thetamax(180)   # degrees -> π
        ax.set_rmax(np.pi/2)   # maximum radius = 90° elevation

        # Title and colorbar
        ax.set_title(f"Polar Path Gain, mode '{ris.mode}', opt '{self.last_optimizer}'", va='bottom', y=1.20)
        cbar = plt.colorbar(pcm, ax=ax, pad=0.2)
        cbar.set_label("Path Gain (dB)")

        plt.show()

    def plot_phase_bf_angle_3d(self, tx_antenna, ris, r, Nr):
        """
        Plot the 3D normalized phase beamforming pattern of a RIS.

        This function evaluates the RIS beamforming gain over a spherical surface of 
        radius `r` centered at the RIS. The response is computed for a grid of azimuth (φ) 
        and elevation (θ) angles, whose coverage depends on the RIS boresight direction 
        ('x', 'y', or 'z'). The resulting pattern is normalized with respect to the 
        maximum theoretical gain, and represented in 3D where the radial distance 
        is proportional to the normalized gain.

        For visualization, the RIS plane is also rendered as a red opaque surface to 
        provide spatial reference with respect to the beamforming pattern.

        Parameters
        ----------
        tx_antenna : object
            Transmitting antenna instance used to compute the RIS beamforming response.

        ris : object
            RIS instance containing attributes such as:
            - `center` (3D coordinates of the RIS center),
            - `boresight` (orientation axis: 'x', 'y', or 'z'),
            - `positions` (element coordinates),
            - `Nrow`, `Ncol` (RIS dimensions).

        r : float
            Radius of the spherical observation surface (distance from the RIS center 
            to the sampling grid).

        Nr : int
            Number of sampling points for both azimuth and elevation angles 
            (controls the angular resolution).

        Returns
        -------
        None
            The function generates and renders an interactive 3D surface plot using Plotly.
            No values are returned; the visualization is directly displayed.
        """
        ris_center = ris.center.copy()
        boresight = ris.boresight.lower()

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
        bf = self.phase_bf_multrx(tx_antenna=tx_antenna, rx_positions=rx_points, ris=ris)
        max_bf = (ris.Nrow * ris.Ncol) ** 2
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
        positions = ris.positions  # shape (Nrow, Ncol, 3)
        v1 = positions[0, -1] - positions[0, 0]  # horizontal axis
        v2 = positions[-1, 0] - positions[0, 0]  # vertical axis

        # Create mesh for the plane
        s = np.linspace(-0.5, 0.5, 2)
        t = np.linspace(-0.5, 0.5, 2)
        S, T = np.meshgrid(s, t)
        plane = ris.center.reshape(1, 1, 3) + S[..., None] * v1 + T[..., None] * v2

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
        return (f"Tang General channel model")

