import os
import numpy as np
import matplotlib.pyplot as plt

# Importamos el optimizador creado anteriormente
from surpRISe.optimization.wmmse_mm import WMMSE_MM_Optimizer, quantize_phases

# Intentamos importar las clases nativas de tu repositorio surpRISe
try:
    from surpRISe.components.antenna import Antenna
    from surpRISe.components.ris import RIS
    from surpRISe.channels.tx2ris2rx.tang_general_channel import TangGeneralChannel
    SURPRISE_NATIVE_AVAILABLE = True
except ImportError:
    SURPRISE_NATIVE_AVAILABLE = False
    print("[AVISO] Usando el motor analítico integrado de Tang General (Near-Field).")


# =====================================================================
# 1. CONFIGURACIÓN DEL EXPERIMENTO (MODIFICA ESTOS PARÁMETROS)
# =====================================================================
FREQ_HZ = 28e9                 # Frecuencia de operación FR2 (28 GHz)
WAVELENGTH = 3e8 / FREQ_HZ     # Longitud de onda (~10.7 mm)

# Geometría de la RIS
NX, NY = 16, 16                # Filas y columnas de la RIS
N_ELEMENTS = NX * NY           # Total de elementos (256)
DX = WAVELENGTH / 2            # Espaciado en X (m)
DY = WAVELENGTH / 2            # Espaciado en Y (m)
RIS_BITS = 2                   # Bits de cuantización (1 bit: 0/180°, 2 bits: 0/90/180/270°)

# Potencia y ruido
TX_POWER_W = 1.0               # Potencia de transmisión (1 W = 30 dBm)
NOISE_POWER_W = 1e-10          # Potencia de ruido (-70 dBm)
NUM_TX_ANTENNAS = 1            # 1 si usas una bocina (Horn) en laboratorio; >1 si usas array

# Coordenadas 3D en metros [x, y, z] asumiendo el centro de la RIS en [0, 0, 0]
# (La RIS está en el plano XZ mirando hacia el eje +Y)
TX_POS = np.array([0.0, 1.2, 0.0])  # Transmisor en campo cercano a 1.2 m

USERS_POS = {
    "Usuario_1": np.array([-0.6, 1.5,  0.2]),  # Izquierda, 1.5 m, ligeramente arriba
    "Usuario_2": np.array([ 0.0, 1.8, -0.1]),  # Centro, 1.8 m, ligeramente abajo
    "Usuario_3": np.array([ 0.7, 1.4,  0.0]),  # Derecha, 1.4 m, plano central
}

# Pesos de prioridad para la prueba general multiusuario (al final del experimento)
USER_WEIGHTS = np.array([1.0, 1.0, 1.0])

# Carpeta donde se guardarán las matrices de estados para el hardware
OUTPUT_DIR = "resultados_experimento_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# 2. MODELO DE CANAL EN CAMPO CERCANO (TANG GENERAL)
# =====================================================================
def get_ris_element_positions(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    """Genera las coordenadas 3D [x, 0, z] de cada celda de la RIS centrada en el origen."""
    x_coords = (np.arange(nx) - (nx - 1) / 2.0) * dx
    z_coords = (np.arange(ny) - (ny - 1) / 2.0) * dy
    xx, zz = np.meshgrid(x_coords, z_coords)
    yy = np.zeros_like(xx)
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def compute_tang_general_channels(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    elem_pos: np.ndarray,
    wavelength: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula los vectores de canal de campo cercano elemento a elemento
    según el modelo general de Tang (onda esférica + patrón de radiación cos^q).
    Devuelve:
        G   : Canal TX -> RIS de dimensión (N, 1)
        h_r : Canal RIS -> RX de dimensión (N,)
    """
    k0 = 2 * np.pi / wavelength
    area_elem = dx * dy

    # Vectores desde cada celda de la RIS hacia TX y hacia RX
    vec_tx = tx_pos - elem_pos  # (N, 3)
    vec_rx = rx_pos - elem_pos  # (N, 3)

    d_tx = np.linalg.norm(vec_tx, axis=1)  # Distancia exacta TX -> celda n
    d_rx = np.linalg.norm(vec_rx, axis=1)  # Distancia exacta celda n -> RX

    # Coseno del ángulo de elevación respecto a la normal de la RIS (eje +Y: [0, 1, 0])
    cos_theta_tx = np.clip(vec_tx[:, 1] / d_tx, 1e-6, 1.0)
    cos_theta_rx = np.clip(vec_rx[:, 1] / d_rx, 1e-6, 1.0)

    # Ganancia efectiva por elemento y propagación esférica
    amp_tx = np.sqrt(area_elem * cos_theta_tx) / (4 * np.pi * d_tx)
    amp_rx = np.sqrt(area_elem * cos_theta_rx) / (4 * np.pi * d_rx)

    G = (amp_tx * np.exp(-1j * k0 * d_tx))[:, None]  # (N, 1)
    h_r = amp_rx * np.exp(-1j * k0 * d_rx)           # (N,)

    return G, h_r


# =====================================================================
# 3. OPTIMIZACIÓN MULTIHAZ WMMSE-MM (PARA 1 ANTENA TX Y K USUARIOS)
# =====================================================================
def optimize_multibeam_wmmse_mm(
    G: np.ndarray,
    H_r: np.ndarray,
    weights: np.ndarray,
    tx_power: float,
    noise_power: float,
    ris_bits: int,
    max_iters: int = 200,
    tol: float = 1e-6,
) -> dict:
    """
    Variante de WMMSE-MM para cuando el laboratorio usa 1 sola bocina transmisora (M=1)
    y queremos que la RIS divida el haz hacia todos los K usuarios simultáneamente
    maximizando la suma ponderada de tasas: sum_k w_k * log2(1 + SNR_k(theta)).
    """
    N, K = H_r.shape
    # Matriz de canales en cascada A: columna k es a_k = diag(h_{r,k}^*) * g
    A = H_r.conj() * G[:, 0:1]  # (N, K), tal que h_eff_k = theta^T @ A[:, k]

    # Inicialización coherente ponderada
    theta = np.exp(-1j * np.angle(A @ weights))
    wsr_history = []

    for _ in range(max_iters):
        # Canal escalar efectivo en cada usuario k: y_k = (theta^H @ a_k^*) s + n_k
        h_eff = np.array([np.vdot(A[:, k], theta) for k in range(K)])  # (K,)
        snr = (tx_power * (np.abs(h_eff) ** 2)) / noise_power
        wsr = float(np.sum(weights * np.log2(1.0 + snr)))
        wsr_history.append(wsr)

        if len(wsr_history) > 1 and abs(wsr_history[-1] - wsr_history[-2]) < tol:
            break

        # Paso WMMSE: receptores escalares u_k y pesos MMSE rho_k
        rx_gain = np.sqrt(tx_power) * h_eff
        u = rx_gain / (np.abs(rx_gain) ** 2 + noise_power)
        mse = np.maximum(1.0 - np.real(u.conj() * rx_gain), 1e-12)
        rho = weights / mse

        # Subproblema MM sobre la RIS: minimizar theta^H U theta - 2 Re(theta^H v)
        U = np.zeros((N, N), dtype=complex)
        v = np.zeros(N, dtype=complex)
        for k in range(K):
            a_k = A[:, k].conj()  # tal que theta^H @ a_k = h_eff_k
            U += rho[k] * (np.abs(u[k]) ** 2) * tx_power * np.outer(a_k, a_k.conj())
            v += rho[k] * (u[k].conj() * np.sqrt(tx_power)) * a_k

        U = 0.5 * (U + U.conj().T)
        lambda_max = float(np.max(np.linalg.eigvalsh(U)))

        # Actualización cerrada Majorization-Minimization (MM)
        for _ in range(30):
            z = (lambda_max * theta - U @ theta) + v
            theta = np.exp(1j * np.angle(z))

    phases_cont = np.mod(np.angle(theta), 2 * np.pi)
    phases_quant = quantize_phases(phases_cont, ris_bits)
    theta_quant = np.exp(1j * phases_quant)

    return {
        "theta_quantized": theta_quant,
        "phases_rad": phases_quant,
        "phases_deg": np.rad2deg(phases_quant),
        "wsr_history": np.array(wsr_history),
    }


# =====================================================================
# 4. FUNCIONES AUXILIARES DE MEDICIÓN Y EXPORTACIÓN A HARDWARE
# =====================================================================
def evaluate_all_users(
    G: np.ndarray, H_r_all: np.ndarray, theta: np.ndarray, tx_power: float, noise_power: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mide la potencia recibida (dBm) y la tasa (bps/Hz) en TODOS los usuarios
    para una configuración fija `theta` de la RIS.
    """
    K = H_r_all.shape[1]
    rx_power_dbm = np.zeros(K)
    rates_bps_hz = np.zeros(K)

    for k in range(K):
        # Ganancia compleja del enlace TX -> RIS -> Usuario k
        h_eff_k = np.sum(H_r_all[:, k] * theta * G[:, 0])
        p_rx_watts = tx_power * (np.abs(h_eff_k) ** 2)
        rx_power_dbm[k] = 10 * np.log10(max(p_rx_watts, 1e-20) / 1e-3)
        snr_k = p_rx_watts / noise_power
        rates_bps_hz[k] = np.log2(1.0 + snr_k)

    return rx_power_dbm, rates_bps_hz


def save_ris_hardware_state(phases_deg: np.ndarray, nx: int, ny: int, n_bits: int, filename: str):
    """Guarda la matriz de estados digitales (0, 1, ..., 2^B - 1) lista para la placa RIS."""
    step = 360.0 / (2 ** n_bits)
    states_2d = (np.round(phases_deg / step).astype(int) % (2 ** n_bits)).reshape((nx, ny))
    filepath = os.path.join(OUTPUT_DIR, filename)
    np.savetxt(filepath, states_2d, fmt="%d", delimiter=",")
    return states_2d, filepath


# =====================================================================
# 5. EJECUCIÓN PRINCIPAL DEL EXPERIMENTO 1
# =====================================================================
def main():
    elem_pos = get_ris_element_positions(NX, NY, DX, DY)
    user_names = list(USERS_POS.keys())
    K = len(user_names)

    # 1. Calculamos los canales de campo cercano para cada usuario
    H_r_list = []
    G = None
    for name in user_names:
        G_k, h_r_k = compute_tang_general_channels(
            TX_POS, USERS_POS[name], elem_pos, WAVELENGTH, DX, DY
        )
        G = G_k
        H_r_list.append(h_r_k)

    H_r_all = np.column_stack(H_r_list)  # Matriz (N, K)

    optimizer = WMMSE_MM_Optimizer(
        num_ris_elements=N_ELEMENTS,
        num_tx_antennas=NUM_TX_ANTENNAS,
        tx_power_max=TX_POWER_W,
        noise_power=NOISE_POWER_W,
        ris_bits=RIS_BITS,
    )

    # Diccionarios para almacenar resultados y graficar
    phase_maps_deg = {}
    power_matrix_dbm = np.zeros((K + 1, K))  # Filas: experimentos, Columnas: usuarios medidos
    rate_matrix = np.zeros((K + 1, K))

    print("=" * 75)
    print(" INICIANDO EXPERIMENTO 1: CAMPO CERCANO (TANG GENERAL + WMMSE-MM/SCRPA)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # PARTE A: MEDICIONES INDIVIDUALES (USUARIO POR USUARIO)
    # -----------------------------------------------------------------
    print("\n>>> FASE 1: APUNTADO Y MEDICIÓN INDIVIDUAL (SINGLE-USER)")
    for idx, name in enumerate(user_names):
        res_su = optimizer.optimize(G=G, H_r=H_r_all[:, idx], method="scrpa")
        theta_su = res_su["theta_quantized"]
        phases_2d = res_su["phases_deg"].reshape((NX, NY))
        phase_maps_deg[f"Individual: {name}"] = phases_2d

        # Guardamos el archivo de estados para cargarlo en el hardware en este paso
        _, fpath = save_ris_hardware_state(
            res_su["phases_deg"], NX, NY, RIS_BITS, f"ris_state_SU_{name}.csv"
        )

        # Medimos en el usuario objetivo y también en el resto de usuarios
        p_dbm, r_bps = evaluate_all_users(G, H_r_all, theta_su, TX_POWER_W, NOISE_POWER_W)
        power_matrix_dbm[idx, :] = p_dbm
        rate_matrix[idx, :] = r_bps

        print(f"\n[Configuración RIS enfocada a {name}] -> Archivo: {fpath}")
        for m_idx, m_name in enumerate(user_names):
            tag = " (OBJETIVO)" if m_idx == idx else ""
            print(f"   Medición en {m_name:10s}: Potencia = {p_dbm[m_idx]:6.2f} dBm | Tasa = {r_bps[m_idx]:5.2f} bps/Hz{tag}")

    # -----------------------------------------------------------------
    # PARTE B: MEDICIÓN GENERAL A TODOS LOS USUARIOS (MULTIUSUARIO)
    # -----------------------------------------------------------------
    print("\n" + "-" * 75)
    print(">>> FASE 2: APUNTADO Y MEDICIÓN GENERAL A TODOS LOS USUARIOS (MULTI-USER)")

    if NUM_TX_ANTENNAS == 1:
        # Si usamos 1 sola bocina TX, usamos WMMSE-MM Multibeam para dividir el haz de la RIS
        res_mu = optimize_multibeam_wmmse_mm(
            G=G,
            H_r=H_r_all,
            weights=USER_WEIGHTS,
            tx_power=TX_POWER_W,
            noise_power=NOISE_POWER_W,
            ris_bits=RIS_BITS,
        )
    else:
        # Si usamos múltiples antenas en TX (M > 1), usamos WMMSE-MM MU-MISO completo
        G_multi = np.repeat(G, NUM_TX_ANTENNAS, axis=1)
        res_mu = optimizer.optimize(
            G=G_multi, H_r=H_r_all, user_weights=USER_WEIGHTS, method="wmmse_mm"
        )

    theta_mu = res_mu["theta_quantized"]
    phases_mu_2d = res_mu["phases_deg"].reshape((NX, NY))
    phase_maps_deg["General: Todos (WMMSE-MM)"] = phases_mu_2d

    _, fpath_mu = save_ris_hardware_state(
        res_mu["phases_deg"], NX, NY, RIS_BITS, "ris_state_MU_Todos.csv"
    )

    p_dbm_mu, r_bps_mu = evaluate_all_users(G, H_r_all, theta_mu, TX_POWER_W, NOISE_POWER_W)
    power_matrix_dbm[K, :] = p_dbm_mu
    rate_matrix[K, :] = r_bps_mu

    print(f"\n[Configuración RIS Multiusuario (WMMSE-MM)] -> Archivo: {fpath_mu}")
    for m_idx, m_name in enumerate(user_names):
        print(f"   Medición en {m_name:10s}: Potencia = {p_dbm_mu[m_idx]:6.2f} dBm | Tasa = {r_bps_mu[m_idx]:5.2f} bps/Hz")
    print(f"   -> Tasa Suma Total (Sum-Rate): {np.sum(r_bps_mu):.2f} bps/Hz")

    # -----------------------------------------------------------------
    # 6. GRÁFICAS DEL EXPERIMENTO
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, K + 1, figsize=(16, 4))
    for ax, (title, p_map) in zip(axes, phase_maps_deg.items()):
        im = ax.imshow(p_map, cmap="twilight", vmin=0, vmax=360)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Elemento X")
        ax.set_ylabel("Elemento Y")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Fase cuantizada (Grados)", shrink=0.8)
    plt.suptitle(f"Mapas de Fase de la RIS en Campo Cercano ({RIS_BITS} bits)", fontsize=13)
    plt.savefig(os.path.join(OUTPUT_DIR, "mapas_fase_experimento_1.png"), dpi=150, bbox_inches="tight")

    # Gráfica comparativa de potencias recibidas
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(K)
    width = 0.2
    labels_exp = [f"Solo {u}" for u in user_names] + ["General (Todos)"]

    for exp_idx in range(K + 1):
        ax2.bar(x_pos + (exp_idx - 1.5) * width, power_matrix_dbm[exp_idx, :], width, label=labels_exp[exp_idx])

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(user_names)
    ax2.set_ylabel("Potencia Recibida (dBm)")
    ax2.set_title("Comparativa: Medición Individual vs. Medición General Multiusuario")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparativa_potencias_exp1.png"), dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()