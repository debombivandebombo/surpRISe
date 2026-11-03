# %% [markdown]
# # Experimento Completo: Estimación de Canal (Tang General), Optimización WMMSE-MM/SCRPA y Envío a RIS Server
# Este cuaderno ejecuta todo el flujo de principio a fin utilizando exactamente el mismo motor de cálculo y envío UDP (`RISConfig` + `RISServerController` a `192.168.100.3:17000`) que `vna_realtime_gui.py`:
# 1. Inicializa la geometría 40x40 de la RIS (`boresight='y'`, dual-pol `'vh'`) y el canal en campo cercano (`surpRISe`).
# 2. Calcula las matrices `state_matrix_v` y `state_matrix_h` para **Usuarios Únicos** y las envía por UDP al servidor de la RIS.
# 3. Calcula las matrices `state_matrix_v` y `state_matrix_h` para **Multiusuario (WMMSE-MM)** y las envía por UDP al servidor de la RIS.

# %%
# =============================================================================
# CELDA 1: IMPORTACIONES Y CLASES DE CONTROL DE LA RIS (IDÉNTICO A vna_realtime_gui.py)
# =============================================================================
import sys
import os
import time
import json
import socket
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Aseguramos que Python encuentre la carpeta 'src' de surpRISe si ejecutamos desde 'notebooks/'
_SURPRISE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(), "..", "src")
)
if os.path.isdir(_SURPRISE_SRC) and _SURPRISE_SRC not in sys.path:
    sys.path.insert(0, _SURPRISE_SRC)

# Importación de los módulos nativos de surpRISe
from surpRISe.components.antenna import Antenna
from surpRISe.components.ris import RIS
from surpRISe.channels.tx2ris2rx.tang_general_channel import tang_general_channel


# =============================================================================
# DEFINICIÓN EXACTA DE RISConfig Y RISServerController (USADA EN vna_realtime_gui.py)
# =============================================================================
@dataclass
class RISConfig:
    state_matrix_v: np.ndarray  # Vertical polarization state matrix (40x40)
    state_matrix_h: np.ndarray  # Horizontal polarization state matrix (40x40)

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
        payload = json.dumps(msg).encode("utf-8")
        self.sock.sendto(payload, self.addr)


print("Modulos de surpRISe y clases RISConfig / RISServerController cargados correctamente.")


# %%
# =============================================================================
# CELDA 2: PARÁMETROS DEL EXPERIMENTO (MISMA CONFIGURACIÓN QUE vna_realtime_gui.py)
# =============================================================================

# --- 1. Conexión UDP con ris_hardware_server.py (Raspberry Pi) ---
RIS_IP = "192.168.100.3"        # <--- IP por defecto de la Raspberry Pi (igual que en vna_realtime_gui.py)
RIS_PORT = 17000                # <--- Puerto UDP de ris_hardware_server.py (17000)
ENVIAR_AL_SERVIDOR = True       # <--- True para enviar los paquetes UDP a la RIS

# --- 2. Parámetros Físicos y de Frecuencia (FR2) ---
C_LIGHT = 299792458.0
FREQ_HZ = 28.0e9                # <--- Frecuencia de operación en Hz (28 GHz)
WAVELENGTH = C_LIGHT / FREQ_HZ  # Longitud de onda (~10.71 mm)

# --- 3. Características de la Placa RIS (40x40, Boresight 'y', Dual-Pol 'vh') ---
RIS_NX = 40                     # Número de columnas de la RIS (Ncol = 40)
RIS_NY = 40                     # Número de filas de la RIS (Nrow = 40)
N_ELEMENTS = RIS_NX * RIS_NY
RIS_DX = WAVELENGTH / 2.0       # Espaciado horizontal entre celdas (delta_h)
RIS_DY = WAVELENGTH / 2.0       # Espaciado vertical entre celdas (delta_v)
RIS_BITS = 1                    # 1 bit por polarización (estados 0 y 1)
RIS_BORESIGHT = "y"             # Eje perpendicular fijo en 'y' (igual que en vna_realtime_gui.py)
OPTIMIZER_SU = "1d_search"      # Optimizador single-user por defecto en vna_realtime_gui.py

# --- 4. Posiciones 3D en el Laboratorio (en metros respecto al centro de la RIS [0, 0, 0]) ---
RIS_POS = np.array([0.0, 0.0, 0.0])       # Posición del centro de la RIS
TX_POS  = np.array([0.0, 1.1, 0.0])       # Posición de la antena transmisora (TX)
TX_GAIN_DBI = 15.0
RX_GAIN_DBI = 15.0

# Diccionario de usuarios receptores (RX): añade, quita o modifica coordenadas [x, y, z]
USUARIOS_RX_POS = {
    "Usuario_1": np.array([-0.5, 1.4,  0.0]),  # Coordenadas 3D del Usuario 1 (m)
    "Usuario_2": np.array([ 0.5, 1.4,  0.0]),  # Coordenadas 3D del Usuario 2 (m)
    "Usuario_3": np.array([ 0.0, 1.7,  0.3]),  # Coordenadas 3D del Usuario 3 (m)
}

# --- 5. Parámetros del Algoritmo WMMSE-MM / SCRPA ---
PESOS_USUARIOS = np.array([1.0, 1.0, 1.0])  # Prioridad de cada usuario en Multiusuario
TX_POWER_W = 1.0                            # Potencia de transmisión de referencia (W)
NOISE_POWER_W = 1e-10                       # Potencia de ruido estimada (W)
MAX_ITER_WMMSE = 50                         # Iteraciones máximas de WMMSE-MM

print(f"Configuracion lista: RIS de {RIS_NY}x{RIS_NX} ({RIS_BITS} bit, boresight='{RIS_BORESIGHT}') a {FREQ_HZ/1e9:.1f} GHz -> UDP {RIS_IP}:{RIS_PORT}")


# %%
# =============================================================================
# CELDA 3: CONSTRUCCIÓN DE GEOMETRÍA Y CANAL TANG GENERAL (SURPRISE)
# =============================================================================

def inicializar_geometria_y_canales(
    tx_pos,
    usuarios_dict,
    ris_nx=RIS_NX,
    ris_ny=RIS_NY,
    freq_hz=FREQ_HZ,
    ris_pos=RIS_POS,
    boresight=RIS_BORESIGHT,
    tx_gain_dbi=TX_GAIN_DBI,
    rx_gain_dbi=RX_GAIN_DBI,
):
    """
    Construye exactamente los mismos objetos RIS, Antenna y tang_general_channel
    que emplea vna_realtime_gui.py (a través de RISCoordinateController).
    """
    lam = C_LIGHT / float(freq_hz)
    k0 = 2.0 * np.pi / lam
    delta_h = lam / 2.0
    delta_v = lam / 2.0
    dx_elem = 0.47 * lam
    dy_elem = 0.47 * lam

    levels = max(2 ** RIS_BITS, 2)
    step_rad = (2.0 * np.pi) / levels
    phase_shifts_dict = {str(i): i * step_rad for i in range(levels)}
    att_dict = {str(i): 1.0 for i in range(levels)}

    ris_obj = RIS(
        center=np.asarray(ris_pos, dtype=float).tolist(),
        boresight=boresight,
        Ncol=int(ris_nx),
        Nrow=int(ris_ny),
        delta_h=delta_h,
        delta_v=delta_v,
        dx=dx_elem,
        dy=dy_elem,
        fc=float(freq_hz),
        radiation_factor=1.0,
        phase_shifts_dict=phase_shifts_dict,
        att_dict=att_dict,
    )
    tx_antenna = Antenna(position=np.asarray(tx_pos, dtype=float).tolist(), gain_dbi=float(tx_gain_dbi))
    channel_model = tang_general_channel()

    elem_coords = ris_obj.positions.reshape(-1, 3)
    normal_vec = ris_obj.normal.copy()

    # Canal TX -> RIS (G_channel)
    vec_tx = np.asarray(tx_pos, dtype=float)[None, :] - elem_coords
    d_tx = np.linalg.norm(vec_tx, axis=1)
    cos_tx = np.clip(np.abs(vec_tx @ normal_vec) / np.maximum(d_tx, 1e-9), 1e-6, 1.0)
    sqrt_g_tx = np.sqrt(10.0 ** (tx_gain_dbi / 10.0))
    G_channel = (
        (sqrt_g_tx * np.sqrt(dx_elem * dy_elem * cos_tx) / (4.0 * np.pi * d_tx))
        * np.exp(-1j * k0 * d_tx)
    )[:, None]

    # Canal RIS -> cada Usuario RX (H_r)
    sqrt_g_rx = np.sqrt(10.0 ** (rx_gain_dbi / 10.0))
    H_r_cols = []
    for _, rx_pos in usuarios_dict.items():
        vec_rx = np.asarray(rx_pos, dtype=float)[None, :] - elem_coords
        d_rx = np.linalg.norm(vec_rx, axis=1)
        cos_rx = np.clip(np.abs(vec_rx @ normal_vec) / np.maximum(d_rx, 1e-9), 1e-6, 1.0)
        h_rk = (
            (sqrt_g_rx * np.sqrt(dx_elem * dy_elem * cos_rx) / (4.0 * np.pi * d_rx))
            * np.exp(-1j * k0 * d_rx)
        )
        H_r_cols.append(h_rk)

    H_r_matriz = np.column_stack(H_r_cols)
    return G_channel, H_r_matriz, ris_obj, tx_antenna, channel_model


G_canal, H_r_canales, ris_inst, tx_inst, canal_tang = inicializar_geometria_y_canales(
    TX_POS, USUARIOS_RX_POS
)
print(f"Canales estimados -> G: {G_canal.shape}, H_r: {H_r_canales.shape} (Boresight RIS: '{ris_inst.boresight}')")


# %%
# =============================================================================
# CELDA 4: CÁLCULO DE ESTADOS 40x40 Y GENERACIÓN DE RISConfig (IDÉNTICO A LA GUI)
# =============================================================================

def cuantizar_fases_ris(fases_rad, n_bits=RIS_BITS):
    """Convierte fases continuas [0, 2*pi) en fases discretas y estados enteros (0..2^B-1)."""
    fases_mod = np.mod(fases_rad, 2.0 * np.pi)
    if n_bits is None or n_bits <= 0:
        return fases_mod, np.zeros_like(fases_mod, dtype=int)
    niveles = 2 ** n_bits
    paso_rad = (2.0 * np.pi) / niveles
    estados_int = np.round(fases_mod / paso_rad).astype(int) % niveles
    fases_cuant_rad = estados_int * paso_rad
    return fases_cuant_rad, estados_int


def calcular_config_usuario_unico(
    rx_pos,
    ris_obj=ris_inst,
    tx_antenna=tx_inst,
    channel_model=canal_tang,
    optimizer=OPTIMIZER_SU,
    freq_hz=FREQ_HZ,
    rx_gain_dbi=RX_GAIN_DBI,
    p_tx=TX_POWER_W,
    sigma2=NOISE_POWER_W,
):
    """
    Calcula la matriz 40x40 para 1 usuario exactamente igual que compute_states_for_coordinates()
    en vna_realtime_gui.py: usando channel_model.optimize() de surpRISe y ris_obj.get_all_state_indices().
    """
    t0 = time.perf_counter()
    rx_arr = np.asarray(rx_pos, dtype=float).reshape(1, 3)
    rx_antenna = Antenna(position=rx_arr[0].tolist(), gain_dbi=float(rx_gain_dbi))

    channel_model.optimize(
        optimizer=optimizer,
        mode="disc",
        wave_model="spherical",
        tx_antenna=tx_antenna,
        rx_antenna=rx_antenna,
        ris=ris_obj,
        verbose=False,
        fc=float(freq_hz),
    )

    estados_2d = ris_obj.get_all_state_indices().astype(int)  # (40, 40)
    fases_q_2d = ris_obj.elem_phase_shift.copy()

    pg = float(
        channel_model.path_gain_multrx(
            tx_antenna=tx_antenna,
            rx_antenna=rx_antenna,
            rx_positions=rx_arr,
            ris=ris_obj,
        )[0]
    )
    p_rx_w = p_tx * pg
    p_rx_dbm = float(10.0 * np.log10(max(p_rx_w, 1e-20) / 1e-3))
    wsr = float(np.log2(1.0 + p_rx_w / sigma2))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    config = RISConfig(state_matrix_v=estados_2d, state_matrix_h=estados_2d.copy())

    return {
        "ris_config": config,
        "estados_40x40": config.state_matrix_v,
        "estados_h_40x40": config.state_matrix_h,
        "fases_rad": fases_q_2d.reshape(-1),
        "fases_deg": np.rad2deg(fases_q_2d),
        "p_rx_dbm": p_rx_dbm,
        "historico_wsr": [wsr],
        "latencia_ms": elapsed_ms,
    }


def optimizar_ris_wmmse_mm(
    G,
    H_r,
    ris_obj=ris_inst,
    pesos=None,
    n_bits=RIS_BITS,
    p_tx=TX_POWER_W,
    sigma2=NOISE_POWER_W,
    max_iter=MAX_ITER_WMMSE,
):
    """
    Calcula la configuración óptima Multiusuario mediante WMMSE-MM exactamente como
    en RISCoordinateController (vna_realtime_gui.py) y devuelve el objeto RISConfig.
    """
    t0 = time.perf_counter()
    if H_r.ndim == 1:
        H_r = H_r[:, None]
    N, K = H_r.shape
    pesos_arr = np.ones(K, dtype=float) if pesos is None else np.asarray(pesos, dtype=float)

    # Canal físico en cascada por celda: A[:, k] = (h_{r,k} * g)^*
    A = (H_r * G[:, 0:1]).conj()

    theta = np.exp(1j * np.angle(A @ pesos_arr))
    historico_wsr = []

    for _ in range(max_iter):
        h_eff_vec = np.array([np.vdot(A[:, k], theta) for k in range(K)])
        snr = (p_tx * (np.abs(h_eff_vec) ** 2)) / sigma2
        wsr = float(np.sum(pesos_arr * np.log2(1.0 + snr)))
        historico_wsr.append(wsr)

        if len(historico_wsr) > 1 and abs(historico_wsr[-1] - historico_wsr[-2]) < 1e-6:
            break

        rx_sig = np.sqrt(p_tx) * h_eff_vec
        u = rx_sig / (np.abs(rx_sig) ** 2 + sigma2)
        mse = np.maximum(1.0 - np.real(u.conj() * rx_sig), 1e-12)
        rho = pesos_arr / mse

        w_u = rho * (np.abs(u) ** 2) * p_tx
        B = A * np.sqrt(w_u)[None, :]  # (N, K): U = B @ B^H
        v = A @ (rho * u.conj() * np.sqrt(p_tx))

        gram_k = B.conj().T @ B  # (K, K)
        gram_k = 0.5 * (gram_k + gram_k.conj().T)
        lambda_max = float(np.max(np.linalg.eigvalsh(gram_k)))

        for _ in range(25):
            u_theta = B @ (B.conj().T @ theta)
            z = (lambda_max * theta - u_theta) + v
            theta = np.exp(1j * np.angle(z))

    fases_opt_cont = np.mod(np.angle(theta), 2.0 * np.pi)
    fases_q_rad, estados_int = cuantizar_fases_ris(fases_opt_cont, n_bits)
    estados_2d = estados_int.reshape((RIS_NY, RIS_NX))
    fases_q_2d = fases_q_rad.reshape((RIS_NY, RIS_NX))
    if ris_obj is not None:
        ris_obj.set_all_shift(fases_q_2d, mode="disc")
        estados_2d = ris_obj.get_all_state_indices().astype(int)
    theta_q = np.exp(1j * fases_q_2d.reshape(-1))

    h_eff_vec = np.array([np.vdot(A[:, k], theta_q) for k in range(K)])
    p_rx_w = float(np.mean(p_tx * (np.abs(h_eff_vec) ** 2)))
    p_rx_dbm = float(10.0 * np.log10(max(p_rx_w, 1e-20) / 1e-3))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    config = RISConfig(state_matrix_v=estados_2d, state_matrix_h=estados_2d.copy())

    return {
        "ris_config": config,
        "estados_40x40": config.state_matrix_v,
        "estados_h_40x40": config.state_matrix_h,
        "fases_rad": fases_q_2d.reshape(-1),
        "fases_deg": np.rad2deg(fases_q_2d),
        "p_rx_dbm": p_rx_dbm,
        "historico_wsr": historico_wsr,
        "latencia_ms": elapsed_ms,
    }


print("Funciones de calculo single-user y multiusuario (WMMSE-MM) listas.")


# %%
# =============================================================================
# CELDA 5: CONEXIÓN Y FUNCIÓN DE ENVÍO UDP (IDÉNTICA A apply_ris_from_ui EN LA GUI)
# =============================================================================

ris_server = RISServerController(ip=RIS_IP, port=RIS_PORT)
ris_connected = False


def enviar_config_ris_udp(info_res, etiqueta=""):
    """
    Envía el objeto RISConfig al servidor UDP (192.168.100.3:17000) exactamente
    igual que hace apply_ris_from_ui() en vna_realtime_gui.py:
    1. Envía {"cmd": "ON", "data": null} la primera vez que conecta.
    2. Envía {"cmd": "SET_CONFIG", "data": gw_pattern} con los 3200 estados (V y H).
    """
    global ris_connected
    config = RISConfig(
        state_matrix_v=info_res["estados_40x40"],
        state_matrix_h=info_res["estados_h_40x40"],
    )

    if not ENVIAR_AL_SERVIDOR:
        print(f"   [Simulacion] RISConfig preparado ({etiqueta}) - envio UDP desactivado.")
        return

    ris_server.addr = (RIS_IP, RIS_PORT)
    if not ris_connected:
        ris_server.connect()
        ris_connected = True
    gw_pattern = ris_server.apply_config(config)
    print(
        f"   [UDP OK -> {RIS_IP}:{RIS_PORT}] SET_CONFIG enviado ({len(gw_pattern)} elementos) | "
        f"{etiqueta} | Prx est: {info_res['p_rx_dbm']:.1f} dBm | {info_res['latencia_ms']:.1f} ms"
    )


print(f"Controlador RISServerController apuntando a {RIS_IP}:{RIS_PORT}.")


# %%
# =============================================================================
# CELDA 6: PRUEBA 1 - CONFIGURACIÓN DE LA RIS PARA USUARIOS ÚNICOS (UNO A UNO)
# =============================================================================

resultados_individuales = {}
nombres_usuarios = list(USUARIOS_RX_POS.keys())

print("=================================================================")
print(" INICIANDO PRUEBA 1: USUARIOS UNICOS (SINGLE-USER)")
print("=================================================================")

for idx, nombre_usuario in enumerate(nombres_usuarios):
    rx_coord = USUARIOS_RX_POS[nombre_usuario]
    print(f"\n---> Calculando RISConfig para apuntar a {nombre_usuario} en RX={rx_coord.tolist()} m")

    res_su = calcular_config_usuario_unico(rx_pos=rx_coord)
    resultados_individuales[nombre_usuario] = res_su

    enviar_config_ris_udp(res_su, etiqueta=f"SU {nombre_usuario}")

    input(f"   >>> RIS apuntando a {nombre_usuario}. Realiza tu medida en el VNA y pulsa [ENTER] para continuar...")


# %%
# =============================================================================
# CELDA 7: PRUEBA 2 - CONFIGURACIÓN DE LA RIS PARA MULTIUSUARIO (TODOS A LA VEZ)
# =============================================================================

print("\n=================================================================")
print(" INICIANDO PRUEBA 2: MULTIUSUARIO CONJUNTO (WMMSE-MM)")
print("=================================================================")

res_multiusuario = optimizar_ris_wmmse_mm(
    G=G_canal,
    H_r=H_r_canales,
    ris_obj=ris_inst,
    pesos=PESOS_USUARIOS,
    n_bits=RIS_BITS,
    p_tx=TX_POWER_W,
    sigma2=NOISE_POWER_W,
    max_iter=MAX_ITER_WMMSE,
)

enviar_config_ris_udp(res_multiusuario, etiqueta="MU Todos (WMMSE-MM)")

print("   >>> RIS configurada en modo MULTIUSUARIO. Ya puedes medir con el VNA en todos los puntos.")
print("\n[FIN] Todas las configuraciones han sido calculadas y enviadas directamente a la RIS.")
