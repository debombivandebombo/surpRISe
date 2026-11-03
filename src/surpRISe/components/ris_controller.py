import json
import socket
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RISConfig:
    state_matrix_v: np.ndarray  # Vertical polarization state matrix (i.e. 40x40)
    state_matrix_h: np.ndarray  # Horizontal polarization state matrix (i.e. 40x40)

    def __init__(
        self,
        state_matrix_v: np.ndarray,
        state_matrix_h: Optional[np.ndarray] = None,
    ) -> None:
        self.state_matrix_v = np.asarray(state_matrix_v, dtype=int)
        if state_matrix_h is None:
            self.state_matrix_h = self.state_matrix_v.copy()
        else:
            self.state_matrix_h = np.asarray(state_matrix_h, dtype=int)


class RISServerController:
    def __init__(self, ip: str = "192.168.100.3", port: int = 17000) -> None:
        self.addr = (ip.strip(), int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def connect(self) -> None:
        self._send_command("ON")

    def disconnect(self) -> None:
        self._send_command("OFF")

    @staticmethod
    def build_gw_pattern(config: RISConfig) -> list[int]:
        v_flat = config.state_matrix_v[::-1, :].T.flatten().astype(int)
        h_flat = config.state_matrix_h[::-1, :].T.flatten().astype(int)

        gw_pattern = [0] * (v_flat.size + h_flat.size)
        gw_pattern[0::2] = [int(x) for x in v_flat.tolist()]
        gw_pattern[1::2] = [int(x) for x in h_flat.tolist()]
        return gw_pattern

    def apply_config(self, config: RISConfig) -> list[int]:
        gw_pattern = self.build_gw_pattern(config)
        self._send_command("SET_CONFIG", gw_pattern)
        return gw_pattern

    def _send_command(self, cmd: str, data=None) -> None:
        msg = {"cmd": str(cmd), "data": data}
        self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)
