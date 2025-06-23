import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi

class tx_ris_rx_channel:
    def __init__(self, tx_antenna=None, rx_antenna=None, ris=None,
                 mode='normal', model='Tang'):
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

        if self.mode not in ['normal', 'perfect']:
            raise ValueError("mode must be either 'normal' or 'perfect'")
        
        # Placeholder for RIS-assisted channel computation
        self.h_ris = None           # Complex channel via RIS
        self.optimal_state = None  # Optimal state index matrix (for normal mode)
        self.optimal_shift = None  # Optimal shift matrix (for perfect mode)



    def __repr__(self):
        return (f"tx_ris_rx_channel(model='{self.model}', mode='{self.mode}', "
                f"RIS size={self.ris.Nrow}x{self.ris.Ncol})")
