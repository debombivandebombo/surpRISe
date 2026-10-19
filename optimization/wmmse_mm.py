import numpy as np
from typing import Optional, Tuple, Dict


def quantize_phases(phases_rad: np.ndarray, n_bits: Optional[int] = None) -> np.ndarray:
    """
    Cuantiza un vector o matriz de fases continuas [0, 2*pi) a estados discretos.
    Si n_bits es None o 0, devuelve las fases continuas en [0, 2*pi).
    """
    phases_mod = np.mod(phases_rad, 2 * np.pi)
    if n_bits is None or n_bits <= 0:
        return phases_mod
    num_levels = 2 ** n_bits
    step = (2 * np.pi) / num_levels
    quantized_indices = np.round(phases_mod / step) % num_levels
    return quantized_indices * step


class WMMSE_MM_Optimizer:
    """
    Optimizador conjunto de precodificación activa y reflexión pasiva (RIS)
    basado en WMMSE-MM y SCRPA.
    Soporta Usuario Único (K=1) y Multiusuario (K>1).
    """

    def __init__(
        self,
        num_ris_elements: int,
        num_tx_antennas: int = 1,
        tx_power_max: float = 1.0,
        noise_power: float = 1e-9,
        ris_bits: Optional[int] = None,
    ):
        self.N = num_ris_elements
        self.M = num_tx_antennas
        self.P_max = tx_power_max
        self.sigma2 = noise_power
        self.ris_bits = ris_bits

    def compute_effective_channel(
        self,
        G: np.ndarray,
        H_r: np.ndarray,
        theta: np.ndarray,
        H_d: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calcula la matriz de canal efectivo H_eff de dimensión (M, K),
        donde h_eff_k^H = h_{d,k}^H + theta^T * diag(h_{r,k}^*) * G.
        """
        K = H_r.shape[1]
        if H_d is None:
            H_d = np.zeros((self.M, K), dtype=complex)

        # H_eff[:, k] es el vector columna h_{eff, k} de tamaño (M,)
        H_eff = np.zeros((self.M, K), dtype=complex)
        for k in range(K):
            cascaded_k_H = (H_r[:, k].conj() * theta) @ G  # (M,)
            H_eff[:, k] = H_d[:, k] + cascaded_k_H.conj()
        return H_eff

    def compute_sum_rate(
        self,
        H_eff: np.ndarray,
        W: np.ndarray,
        user_weights: Optional[np.ndarray] = None,
    ) -> float:
        """Calcula la tasa suma ponderada (Weighted Sum-Rate en bps/Hz)."""
        K = H_eff.shape[1]
        if user_weights is None:
            user_weights = np.ones(K)

        wsr = 0.0
        for k in range(K):
            h_k_H = H_eff[:, k].conj().T
            signal = np.abs(h_k_H @ W[:, k]) ** 2
            interference = sum(
                np.abs(h_k_H @ W[:, j]) ** 2 for j in range(K) if j != k
            )
            sinr_k = signal / (interference + self.sigma2)
            wsr += user_weights[k] * np.log2(1.0 + sinr_k)
        return float(wsr)

    def _update_ris_mm(
        self,
        G: np.ndarray,
        H_r: np.ndarray,
        H_d: np.ndarray,
        W: np.ndarray,
        u: np.ndarray,
        rho: np.ndarray,
        theta_curr: np.ndarray,
        mm_inner_iters: int = 50,
        tol_mm: float = 1e-6,
    ) -> np.ndarray:
        """
        Subproblema de actualización de fases de la RIS mediante
        Majorization-Minimization (MM) / SCRPA.
        """
        K = H_r.shape[1]
        U = np.zeros((self.N, self.N), dtype=complex)
        v = np.zeros(self.N, dtype=complex)

        # Construcción de las matrices auxiliares U y v
        for k in range(K):
            Phi_k = H_r[:, k].conj()[:, None] * G  # diag(h_{r,k}^*) @ G -> (N, M)
            for j in range(K):
                a_kj = Phi_k @ W[:, j]  # (N,)
                c_kj = np.vdot(H_d[:, k], W[:, j])  # h_{d,k}^H @ w_j

                U += rho[k] * (np.abs(u[k]) ** 2) * np.outer(a_kj.conj(), a_kj)
                v -= rho[k] * (np.abs(u[k]) ** 2) * c_kj * a_kj.conj()

            a_kk = Phi_k @ W[:, k]
            v += rho[k] * u[k] * a_kk.conj()

        # Cota superior de curvatura (máximo autovalor de la matriz hermítica U)
        U = 0.5 * (U + U.conj().T)
        lambda_max = float(np.max(np.linalg.eigvalsh(U)))

        theta = theta_curr.copy()
        for _ in range(mm_inner_iters):
            z = (lambda_max * theta - U @ theta) + v
            theta_next = np.exp(1j * np.angle(z))
            if np.linalg.norm(theta_next - theta) < tol_mm:
                theta = theta_next
                break
            theta = theta_next

        return theta

    def optimize(
        self,
        G: np.ndarray,
        H_r: np.ndarray,
        H_d: Optional[np.ndarray] = None,
        user_weights: Optional[np.ndarray] = None,
        max_outer_iters: int = 100,
        mm_inner_iters: int = 50,
        tol: float = 1e-4,
        method: str = "wmmse_mm",
    ) -> Dict[str, np.ndarray]:
        """
        Ejecuta la optimización WMMSE-MM o SCRPA.

        Parámetros:
        -----------
        G : np.ndarray de forma (N, M)
            Canal desde el Transmisor (TX) hasta la RIS.
        H_r : np.ndarray de forma (N,) para 1 usuario o (N, K) para K usuarios
            Canal desde la RIS hasta el/los Receptor(es).
        H_d : np.ndarray opcional de forma (M,) o (M, K)
            Canal directo TX-RX (si está bloqueado, se asume 0).
        method : str ("wmmse_mm" o "scrpa")
            Algoritmo seleccionado.
        """
        # Normalización de dimensiones para soportar K=1 (1D) o K>1 (2D)
        if H_r.ndim == 1:
            H_r = H_r[:, None]
        K = H_r.shape[1]

        if H_d is None:
            H_d = np.zeros((self.M, K), dtype=complex)
        elif H_d.ndim == 1:
            H_d = H_d[:, None]

        if user_weights is None:
            user_weights = np.ones(K)

        # Inicialización de fases aleatorias o en cero y precodificador uniforme
        theta = np.ones(self.N, dtype=complex)
        H_eff = self.compute_effective_channel(G, H_r, theta, H_d)

        # Inicialización MRT (Maximum Ratio Transmission) para W
        W = H_eff / (np.linalg.norm(H_eff, "fro") + 1e-12) * np.sqrt(self.P_max)

        wsr_history = [self.compute_sum_rate(H_eff, W, user_weights)]

        # Caso rápido SCRPA para Usuario Único (K=1) y 1 antena TX (M=1)
        if method.lower() == "scrpa" and K == 1 and self.M == 1:
            a_11 = (H_r[:, 0].conj() * G[:, 0])
            ref_phase = np.angle(H_d[0, 0]) if np.abs(H_d[0, 0]) > 0 else 0.0
            opt_phases = np.mod(ref_phase - np.angle(a_11), 2 * np.pi)
            q_phases = quantize_phases(opt_phases, self.ris_bits)
            theta = np.exp(1j * q_phases)
            H_eff = self.compute_effective_channel(G, H_r, theta, H_d)
            wsr_history.append(self.compute_sum_rate(H_eff, W, user_weights))
            return {
                "theta_continuous": np.exp(1j * opt_phases),
                "theta_quantized": theta,
                "phases_rad": q_phases,
                "phases_deg": np.rad2deg(q_phases),
                "W": W,
                "wsr_history": np.array(wsr_history),
            }

        # Bucle principal de optimización alternada WMMSE-MM
        for _ in range(max_outer_iters):
            H_eff = self.compute_effective_channel(G, H_r, theta, H_d)

            # 1. Actualización del receptor MMSE (u_k) y pesos (rho_k)
            u = np.zeros(K, dtype=complex)
            rho = np.zeros(K, dtype=float)

            for k in range(K):
                h_k_H = H_eff[:, k].conj().T
                denom = sum(np.abs(h_k_H @ W[:, j]) ** 2 for j in range(K)) + self.sigma2
                u[k] = (h_k_H @ W[:, k]) / denom
                e_k = max(1.0 - np.real(u[k].conj() * (h_k_H @ W[:, k])), 1e-12)
                rho[k] = user_weights[k] / e_k

            # 2. Actualización del precodificador de transmisión (W)
            if self.M == 1 and K == 1:
                W = np.array([[np.sqrt(self.P_max)]], dtype=complex)
            else:
                A_tx = sum(
                    rho[j] * (np.abs(u[j]) ** 2) * np.outer(H_eff[:, j], H_eff[:, j].conj())
                    for j in range(K)
                )
                # Regularización acoplada a la potencia máxima
                reg = (K * self.sigma2 / self.P_max) * np.eye(self.M)
                W_raw = np.zeros((self.M, K), dtype=complex)
                A_inv = np.linalg.pinv(A_tx + reg)
                for k in range(K):
                    W_raw[:, k] = rho[k] * u[k] * (A_inv @ H_eff[:, k])
                norm_W = np.linalg.norm(W_raw, "fro")
                W = W_raw * (np.sqrt(self.P_max) / max(norm_W, 1e-12))

            # 3. Actualización de las fases de la RIS mediante MM
            theta = self._update_ris_mm(
                G, H_r, H_d, W, u, rho, theta, mm_inner_iters=mm_inner_iters
            )

            # Evaluación de convergencia
            H_eff_new = self.compute_effective_channel(G, H_r, theta, H_d)
            wsr_new = self.compute_sum_rate(H_eff_new, W, user_weights)
            wsr_history.append(wsr_new)

            if abs(wsr_new - wsr_history[-2]) < tol:
                break

        phases_cont = np.mod(np.angle(theta), 2 * np.pi)
        phases_quant = quantize_phases(phases_cont, self.ris_bits)
        theta_quant = np.exp(1j * phases_quant)

        return {
            "theta_continuous": theta,
            "theta_quantized": theta_quant,
            "phases_rad": phases_quant,
            "phases_deg": np.rad2deg(phases_quant),
            "W": W,
            "wsr_history": np.array(wsr_history),
        }