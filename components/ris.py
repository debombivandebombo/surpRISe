import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
import copy

import scienceplots
plt.style.use(['science','no-latex'])
plt.rcParams.update({'font.size': 15})

class RIS:
    def __init__(self,
                 center=[0, 0, 0],          # (3,) Center of the RIS in 3D (default at origin)
                 boresight='z',            # 'x', 'y', or 'z' (default 'z')
                 Ncol=None, Nrow=None,     # Number of columns and rows (required)
                 delta_h=None, delta_v=None,  # Spacing between elements (required)
                 phase_shifts_dict=None,   # dict {int: phase_shift in radians per state}
                 att_dict=None,            # dict {int: attenuation value per state}
                 dx=None, dy=None,         # Element width and height (required)
                 fc=None,                  # Operating frequency (Hz) (required)
                 radiation_factor=1):      # Cosine pattern exponent (default = 1)
        """
        Represents a Reconfigurable Intelligent Surface (RIS) in a 3D environment.

        This class models a rectangular RIS panel composed of discrete reflecting elements
        arranged in rows and columns. Each element can be configured with a specific phase shift
        and attenuation, defined per state. The panel orientation is defined by its center point
        and a fixed boresight axis ('x', 'y', or 'z'). The class also stores geometric and 
        electromagnetic properties relevant to beamforming and wave reflection modeling.

        Parameters
        ----------
        center : list of float, optional (default = [0, 0, 0])
            3D coordinates of the RIS center in meters.

        boresight : str, optional (default = 'z')
            Normal direction of the RIS surface. Must be one of 'x', 'y', or 'z'.

        Ncol : int
            Number of columns (horizontal elements) in the RIS panel.

        Nrow : int
            Number of rows (vertical elements) in the RIS panel.

        delta_h : float
            Horizontal spacing between adjacent elements (in meters).

        delta_v : float
            Vertical spacing between adjacent elements (in meters).

        phase_shifts_dict : dict
            Dictionary mapping each RIS state (int) to a phase shift value (in radians).
            For example: `{0: 0.0, 1: pi/2, 2: pi}`.

        att_dict : dict
            Dictionary mapping each RIS state (int) to an attenuation factor (linear scale).
            For example: `{0: 1.0, 1: 0.9, 2: 0.7}`.

        dx : float
            Width of each RIS element (in meters).

        dy : float
            Height of each RIS element (in meters).

        fc : float
            Operating frequency of the RIS in Hz.

        radiation_factor : float, optional (default = 1)
            Exponent of the cosine-based radiation pattern model. A value of 1 corresponds
            to a classical cosine pattern; higher values narrow the beam.

        Attributes
        ----------
        positions : ndarray
            3D coordinates of each RIS element, shaped (Nrow, Ncol, 3).

        normal : ndarray
            Unit vector representing the RIS normal (boresight) in 3D.

        N_elements: int
            Total number of elements of the RIS (Nrow * Ncol).

        mode: str
            Indicates discrete ('disc') or continuous/perfect ('cont') modes of operating 
            the shifts of the RIS.

        perfect_phase_enabled : bool
            Flag indicating discrete (False) or continuous mode (True).

        Notes
        -----
        This class assumes a static, planar RIS with discrete, predefined states, but can 
        also work in continuous (perfect) mode.
        """
        # Basic parameters
        self.center = np.array(center, dtype=float)
        self.boresight = boresight.lower()
        self.Ncol = Ncol
        self.Nrow = Nrow
        self.delta_h = delta_h
        self.delta_v = delta_v
        self.dx = dx
        self.dy = dy
        self.fc = fc
        self.radiation_factor = radiation_factor

        # Validations for required parameters
        for var_name in ['Ncol', 'Nrow', 'delta_h', 'delta_v', 'dx', 'dy', 'fc']:
            if eval(var_name) is None:
                raise ValueError(f"{var_name} must be provided and not None.")
        
        if not isinstance(phase_shifts_dict, dict):
            raise TypeError("phase_shifts_dict must be a dictionary")
        if not isinstance(att_dict, dict):
            raise TypeError("att_dict must be a dictionary")
        if set(phase_shifts_dict.keys()) != set(att_dict.keys()):
            raise ValueError("phase_shifts_dict and att_dict must have the same keys (states)")
        
        self.phase_shifts_dict = self._parse_dict_keys_to_int(phase_shifts_dict)
        self.att_dict = self._parse_dict_keys_to_int(att_dict)
        
        if self.boresight not in ('x', 'y', 'z'):
            raise ValueError("Boresight must be one of 'x', 'y', or 'z'")
        
        self.N_elements = self.Ncol * self.Nrow
        self.normal = self._get_normal_from_boresight()
        self.positions = self._generate_positions()
        self.perfect_phase_enabled = False
        self.mode = 'disc'

        self.reset_ris()
     
    # Functions:
    def _get_normal_from_boresight(self):
        if self.boresight == 'x':
            return np.array([1, 0, 0])
        elif self.boresight == 'y':
            return np.array([0, 1, 0])
        elif self.boresight == 'z':
            return np.array([0, 0, 1])

    def _generate_positions(self):
        positions = np.zeros((self.Nrow, self.Ncol, 3), dtype=float)

        row_indices = np.arange(self.Nrow)
        col_indices = np.arange(self.Ncol)

        row_offsets = (row_indices - (self.Nrow - 1) / 2) * self.delta_v
        col_offsets = (col_indices - (self.Ncol - 1) / 2) * self.delta_h

        for i, r_off in enumerate(row_offsets):
            for j, c_off in enumerate(col_offsets):
                if self.boresight == 'x':
                    # Plane YZ: +Y (left to right), -Z (top to bottom)
                    pos = self.center + np.array([0, c_off, -r_off])
                elif self.boresight == 'y':
                    # Plane XZ: -X (left to right), -Z (top to bottom)
                    pos = self.center + np.array([-c_off, 0, -r_off])
                elif self.boresight == 'z':
                    # Plane XY: +X (left to right), -Y (top to bottom)
                    pos = self.center + np.array([c_off, -r_off, 0])
                positions[i, j, :] = pos

        return positions
    
    def _parse_dict_keys_to_int(self, d):
        return {int(k): v for k, v in d.items()}
    
    def reset_ris(self):
        """Initialize elem_state, elem_phase_shift and elem_att with the fist state values."""
        first_state = sorted(self.phase_shifts_dict.keys())[0]
        
        # elem_state: matrix with the fist state number
        self.elem_state = np.full((self.Nrow, self.Ncol), first_state, dtype=int)
        
        # elem_phase_shift: matrix with the phase_shift from the first state
        phase_shift_value = self.phase_shifts_dict[first_state]
        self.elem_phase_shift = np.full((self.Nrow, self.Ncol), phase_shift_value, dtype=float)
        
        # elem_att: matrix with the elem_att of the first state
        att_value = self.att_dict[first_state]
        self.elem_att = np.full((self.Nrow, self.Ncol), att_value, dtype=float)

    def set_all_state(self, state_matrix):
        """
        Updates elem_state, elem_phase_shift, and elem_att according to the given state matrix.

        Parameters:
        -----------
        state_matrix : ndarray of int, shape (Nrow, Ncol)
            Matrix with the desired state for each RIS element.
        """
        state_matrix = np.array(state_matrix, dtype=int)
        
        if state_matrix.shape != (self.Nrow, self.Ncol):
            raise ValueError(f"state_matrix must have shape ({self.Nrow}, {self.Ncol})")
        
        unique_states = np.unique(state_matrix)

        for st in unique_states:
            if st not in self.phase_shifts_dict:
                print(st)
                raise ValueError(f"State {st} not found in phase_shifts_dict")
            if st not in self.att_dict:
                raise ValueError(f"State {st} not found in att_dict")
        
        self.elem_state = state_matrix.copy()
        
        # Update elem_phase_shift and elem_att according to elem_state
        get_phase = np.vectorize(self.phase_shifts_dict.get, otypes=[float])
        get_att   = np.vectorize(self.att_dict.get, otypes=[float])

        self.elem_phase_shift = get_phase(self.elem_state)
        self.elem_att = get_att(self.elem_state)

        self.perfect_phase_enabled = False
        self.mode = 'disc'

    def set_all_shift(self, shift_matrix, mode='disc'):
        """
        Updates RIS element configuration based on a matrix of phase shifts.

        Parameters:
        -----------
        shift_matrix : ndarray or list of float, shape (Nrow, Ncol)
            Desired phase shift (in radians) for each element.
        mode : str
            'disc' to use predefined discrete states from phase_shifts_dict,
            'cont' to apply arbitrary continuous phase shifts directly.
        """
        shift_matrix = np.array(shift_matrix, dtype=float)

        if shift_matrix.shape != (self.Nrow, self.Ncol):
            raise ValueError(f"shift_matrix must have shape ({self.Nrow}, {self.Ncol})")

        if mode == 'cont':
            self.elem_phase_shift = shift_matrix
            self.elem_att = np.ones((self.Nrow, self.Ncol), dtype=float)
            self.elem_state = np.full((self.Nrow, self.Ncol), -1, dtype=int)
            self.perfect_phase_enabled = True
            self.mode = 'cont'

        elif mode == 'disc':

            # Map from shift value to state index
            shift_to_state = {v: k for k, v in self.phase_shifts_dict.items()}

            # Create output arrays
            state_matrix = np.full_like(shift_matrix, -999, dtype=int)

            for shift_val in np.unique(shift_matrix):
                if shift_val not in shift_to_state:
                    raise ValueError(f"Phase shift value {shift_val} is not in phase_shifts_dict")
                state_matrix[shift_matrix == shift_val] = shift_to_state[shift_val]

            self.set_all_state(state_matrix)
            self.perfect_phase_enabled = False
            self.mode = 'disc'

        else:
            raise ValueError("mode must be either 'disc' or 'cont'")

    def get_all_phase_shifts(self):
        """Returns a (Nrow, Ncol) array of phase shifts in radians."""
        return self.elem_phase_shift.copy()

    def get_all_state_indices(self):
        """Returns a (Nrow, Ncol) array of state indices."""
        return self.elem_state.copy()

    def get_all_refl_coef(self):
        """Returns a (Nrow, Ncol) array of complex weights: attenuation * exp(1j * phase)."""
        return self.elem_att * np.exp(1j * self.elem_phase_shift)
    
    def get_rad_pat(self, position_xyz):
        """
        Compute the RIS radiation pattern for all elements toward a given 3D position.

        Parameters:
        -----------
        position_xyz : ndarray of shape (3,)
            Target 3D position to evaluate the radiation pattern toward.

        Returns:
        --------
        rad_pat : ndarray of shape (Nrow, Ncol)
            Radiation pattern from each RIS element toward the given position.
        """
        # Vector from each RIS element to the target position
        direction_vectors = position_xyz - self.positions  # shape: (Nrow, Ncol, 3)

        # Normalize direction vectors
        norms = np.linalg.norm(direction_vectors, axis=2, keepdims=True)
        direction_unit = direction_vectors / norms  # shape: (Nrow, Ncol, 3)

        # Compute angle between direction and RIS normal
        # self.normal is assumed already normalized
        cos_ang = np.sum(direction_unit * self.normal, axis=2)  # shape: (Nrow, Ncol)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)  # Ensure valid range
        ang_rel = np.arccos(cos_ang)

        # Compute radiation pattern
        rad_pat = np.zeros_like(ang_rel)
        valid = (ang_rel >= 0) & (ang_rel <= np.pi / 2)
        rad_pat[valid] = np.cos(ang_rel[valid]) ** self.radiation_factor

        return rad_pat  # shape: (Nrow, Ncol)

    def plot_phase_dist(self):
        """Plots the 2D distribution of phase shifts using matplotlib."""
        wrapped_phase = np.mod(self.elem_phase_shift, 2 * np.pi)
        
        plt.figure(figsize=(6, 5))
        mesh = plt.pcolormesh(np.arange(self.Ncol+1)-0.5,
                              np.arange(self.Nrow+1)-0.5,
                              wrapped_phase,
                              cmap='viridis', shading='auto',
                              vmin=0, vmax=2*np.pi)
        cbar = plt.colorbar(mesh, label=r'Phase Shift, $\psi$ (rad)')
        ticks = np.arange(0, 2*np.pi + 0.1, np.pi/2)  # +0.1 para incluir 2π por redondeo
        tick_labels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.title(f"RIS Phase Distribution")
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.xticks(np.arange(self.Ncol))
        plt.yticks(np.arange(self.Nrow))
        # plt.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.gca().set_xticks(np.arange(-0.5, self.Ncol, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, self.Nrow, 1), minor=True)
        plt.grid(True, which='minor', linestyle='--', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def export_pattern_to_csv(self, filename):
        """
        Export elem_state to a CSV-style string with comma-separated rows,
        like [[-1 -1],[-1 -1]], all in one line.

        Parameters:
        -----------
        filename : str
            Name of the filename to be saved.

        Returns:
        --------
        None
            Saves the desired filename.
        """
        if not filename.lower().endswith('.csv'):
            filename += '.csv'

        try:
            rows = []
            for row in self.elem_state:
                row_str = '[' + ' '.join(map(str, row)) + ']'
                rows.append(row_str)
            array_str = '[' + ','.join(rows) + ']'

            with open(filename, 'w') as f:
                f.write(array_str)

            print(f"Element state matrix exported to {filename} in compact format.")
        except Exception as e:
            print(f"Error exporting RIS element state: {e}")

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f"RIS({self.Nrow}x{self.Ncol}, boresight='{self.boresight}', center={self.center.tolist()})"
    
