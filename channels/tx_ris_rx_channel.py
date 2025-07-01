import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from optimizers.tx_ris_rx_opt import tx_ris_rx_direct_disc_opt, tx_ris_rx_max_bf_opt

import scienceplots
plt.style.use(['science','no-latex'])
plt.rcParams.update({'font.size': 15})

class tx_ris_rx_channel:
    def __init__(self, tx_antenna=None, rx_antenna=None, ris=None,
                 mode='normal', model='Tang_general', optimizer='max_bf'):
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
            Channel model name (e.g., 'Tang_general', 'Tang_farfield', etc.).
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

        # RIS phase shift exponential contribution
        exp_ris = np.exp(1j * phase_shift_matrix)

        for i in range(Nrx):
            rx_pos = rx_positions[i]
            d_ris_rx = np.linalg.norm(ris_pos - rx_pos, axis=2)
            exp_ris_rx = np.exp(-1j * k * d_ris_rx)

            # Element-wise product and sum
            exp_total = exp_tx_ris * exp_ris * exp_ris_rx
            phase_sums[i] = np.sum(exp_total)

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
        ris_pos = self.ris.positions  # shape: (Nrow, Ncol, 3)
        tx_pos = self.tx_antenna.position  # shape: (3,)

        # RIS phase shift matrix
        phase_shift_matrix = self.ris.elem_phase_shift  # shape: (Nrow, Ncol)

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

        # Distance and phase from Tx to RIS
        d_tx_ris = np.linalg.norm(ris_pos - tx_pos, axis=2)
        exp_tx_ris = np.exp(-1j * k * d_tx_ris)

        # RIS reflection coefficient (Nrow x Ncol complex matrix)
        exp_ris = self.ris.get_all_refl_coef()

        # Placeholder for path gains per receiver
        path_gains = np.zeros(Nrx)

        for i in range(Nrx):
            rx_pos = rx_positions[i]

            # Distance and phase from RIS to current Rx
            d_ris_rx = np.linalg.norm(ris_pos - rx_pos, axis=2)
            exp_ris_rx = np.exp(-1j * k * d_ris_rx)

            # Total complex phase response per element
            exp_total = exp_tx_ris * exp_ris * exp_ris_rx

            # Placeholder for directional gains of
            # ris_rad_pat_tx and ris_rad_pat_rx
            # both of shape (Nrow, Ncol)
            ris_rad_pat_tx = self.ris.get_rad_pat(tx_pos)
            ris_rad_pat_rx = self.ris.get_rad_pat(rx_pos)

            # Element-wise product: combined radiation pattern
            f_combine = ris_rad_pat_tx * ris_rad_pat_rx     # Radiation pattern of Tx and Rx antennas assumed to be 1 in all the RIS elements for the moment

            # Path gain for this Rx
            path_gains[i] = gain_constant * np.abs(
                np.sum(exp_total * np.sqrt(f_combine) / (d_tx_ris * d_ris_rx))
            ) ** 2

        return path_gains
    
    def plot_phase_bf_2d_map(self, ax1lim, ax2lim, Nax1, Nax2, plane='xz'):
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
        plt.title(f"Phase beamforming distribution in {plane.upper()} plane, mode '{self.mode}', opt '{self.optimizer}'")
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
        plt.title(f"Path gain distribution in {plane.upper()} plane, mode '{self.mode}', opt '{self.optimizer}'")
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
        plt.xlabel(f"Angle (deg) in {axis.upper()}-{['x','y','z'][normal_idx].upper()} plane")
        plt.ylabel("Normalized Beamforming")
        plt.title(f"Phase beamforming vs Angle, mode '{self.mode}', opt '{self.optimizer}'")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_path_gain_angle(self, r, Nr, axis='x', vmin=-160, vmax=-80):
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
        plt.xlabel(f"Angle (deg) in {axis.upper()}-{['x','y','z'][normal_idx].upper()} plane")
        plt.ylabel("Path Gain (dB)")
        plt.title(f"Path Gain vs Angle, mode '{self.mode}', opt '{self.optimizer}'")
        plt.ylim([vmin, vmax])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_phase_bf_angle_2d_map(self, r, Nr):
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
        plt.xlabel('Azimuth φ (deg)')
        plt.ylabel('Elevation θ (deg)')
        plt.title(f"2D Angular Phase Beamforming Map, mode '{self.mode}', opt '{self.optimizer}'")
        plt.colorbar(pcm, label='Normalized Beamforming')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_path_gain_angle_2d_map(self, r, Nr, vmin=-160, vmax=-80):
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
        plt.xlabel('Azimuth φ (deg)')
        plt.ylabel('Elevation θ (deg)')
        plt.title(f"2D Angular Path Gain Map, mode '{self.mode}', opt '{self.optimizer}'")
        plt.colorbar(pcm, label='Path Gain (dB)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # def plot_phase_bf_angle_3d(self, r=10, Nr=180):
    #     """
    #     Plot the 3D beamforming response (normalized) as a spherical diagram centered at the origin.

    #     Parameters:
    #     ------------
    #     r : float
    #         Radius at which receiver points are placed.
    #     Nr : int
    #         Resolution for azimuth and elevation angles.
    #     """
    #     ris_center = self.ris.center.copy()
    #     boresight = self.ris.boresight.lower()

    #     if boresight == 'z':
    #         theta = np.linspace(0, pi / 2, Nr)  # elevation
    #         phi = np.linspace(-pi, pi, Nr)   # azimuth
    #     elif boresight == 'x':
    #         theta = np.linspace(0, pi, Nr)  # elevation
    #         phi = np.linspace(-pi / 2, pi / 2, Nr)   # azimuth
    #     else:   # boresight == 'y'
    #         theta = np.linspace(0, pi, Nr)  # elevation
    #         phi = np.linspace(0, pi, Nr)   # azimuth

    #     Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

    #     x = r * np.sin(Theta) * np.cos(Phi)
    #     y = r * np.sin(Theta) * np.sin(Phi)
    #     z = r * np.cos(Theta)

    #     rx_points = r * np.stack([x, y, z], axis=-1) + ris_center  # (Nr, Nr, 3)
    #     rx_points = rx_points.reshape(-1, 3)

    #     bf = self.phase_bf_multrx(rx_points)
    #     max_bf = (self.ris.Nrow * self.ris.Ncol)**2
    #     R = bf.reshape(Nr, Nr) / max_bf # Beamforming normalized between 0 and 1

    #     X = R * x
    #     Y = R * y
    #     Z = R * z

    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_surface(X, Y, Z,
    #                            rstride=1, cstride=1, antialiased=True, alpha=0.9)

    #     m = plt.cm.ScalarMappable(cmap='viridis')
    #     m.set_array(R)
    #     m.set_clim(0, 1)
    #     fig.colorbar(m, shrink=0.6, aspect=10, label='Normalized BF', ax=ax)

    #     ax.set_title("3D Beamforming Pattern (Normalized)")
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_box_aspect([1, 1, 1])
    #     plt.tight_layout()
    #     plt.show()

    def __repr__(self):
        return (f"tx_ris_rx_channel(model='{self.model}', mode='{self.mode}', "
                f"RIS size={self.ris.Nrow}x{self.ris.Ncol})")
