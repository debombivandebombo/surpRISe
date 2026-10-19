# %% [markdown]
# # Experimento Completo: Estimación de Canal (Tang General), Optimización WMMSE-MM/SCRPA y Envío a RIS Server
# Este cuaderno ejecuta todo el flujo de principio a fin:
# 1. Estima el canal en campo cercano usando los módulos del proyecto (`surpRISe`).
# 2. Calcula las fases de la RIS para **Usuarios Únicos** y las envía al servidor de la RIS.
# 3. Calcula las fases de la RIS para **Multiusuario** y las envía al servidor de la RIS.

# %%
# =============================================================================
# CELDA 1: IMPORTACIONES DEL PROYECTO Y LIBRERÍAS ESTÁNDAR
# =============================================================================
import sys
import os
import json
import socket
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# Aseguramos que Python encuentre la carpeta 'src' de surpRISe si ejecutamos desde 'notebooks/'
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

# Importación de los módulos existentes en el proyecto surpRISe
from surpRISe.components.antenna import Antenna
from surpRISe.components.ris import RIS
from surpRISe.channels.tx2ris2rx.tang_general_channel import TangGeneralChannel

print("✓ Módulos cargados correctamente.")


# %%
# =============================================================================
# CELDA 2: PARÁMETROS DEL EXPERIMENTO (MODIFICA ESTOS VALORES SEGÚN TU SETUP)
# =============================================================================

# --- 1. Conexión con ris_hardware_server.py ---
SERVER_IP = "127.0.0.1"         # <--- [MODIFICABLE] IP donde corre ris_hardware_server.py (ej. "192.168.1.100")
SERVER_PORT = 8000              # <--- [MODIFICABLE] Puerto configurado en ris_hardware_server.py
SERVER_PROTOCOL = "http"        # <--- [MODIFICABLE] "http" (REST/JSON) o "socket" (TCP raw) según tu servidor
SERVER_ENDPOINT = "/set_ris"    # <--- [MODIFICABLE] Ruta del endpoint si tu servidor usa HTTP (ej. "/configure" o "/")
ENVIAR_AL_SERVIDOR = True       # <--- [MODIFICABLE] Pon False si quieres probar el código sin estar conectado a la RIS

# --- 2. Parámetros Físicos y de Frecuencia (FR2) ---
FREQ_HZ = 28e9                  # <--- [MODIFICABLE] Frecuencia de operación en Hz (28 GHz)
WAVELENGTH = 3e8 / FREQ_HZ      # Longitud de onda calculada automáticamente (~10.71 mm)

# --- 3. Características de la Placa RIS ---
RIS_NX = 16                     # <--- [MODIFICABLE] Número de elementos de la RIS en el eje X (columnas)
RIS_NY = 16                     # <--- [MODIFICABLE] Número de elementos de la RIS en el eje Y (filas)
RIS_DX = WAVELENGTH / 2         # <--- [MODIFICABLE] Espaciado entre celdas en X (en metros)
RIS_DY = WAVELENGTH / 2         # <--- [MODIFICABLE] Espaciado entre celdas en Y (en metros)
RIS_BITS = 1                    # <--- [MODIFICABLE] Bits de fase de la RIS (1 bit = 2 estados, 2 bits = 4 estados)

# --- 4. Posiciones 3D en el Laboratorio (Campo Cercano) ---
# Coordenadas [x, y, z] en metros respecto al centro de la RIS [0, 0, 0]
RIS_POS = np.array([0.0, 0.0, 0.0])       # <--- [MODIFICABLE] Posición del centro de la RIS
TX_POS  = np.array([0.0, 1.1, 0.0])       # <--- [MODIFICABLE] Posición de la antena transmisora (TX)

# Diccionario de usuarios receptores (RX): añade, quita o modifica coordenadas [x, y, z]
USUARIOS_RX_POS = {
    "Usuario_1": np.array([-0.5, 1.4,  0.0]),  # <--- [MODIFICABLE] Coordenadas 3D del Usuario 1 (m)
    "Usuario_2": np.array([ 0.5, 1.4,  0.0]),  # <--- [MODIFICABLE] Coordenadas 3D del Usuario 2 (m)
    "Usuario_3": np.array([ 0.0, 1.7,  0.3]),  # <--- [MODIFICABLE] Coordenadas 3D del Usuario 3 (m)
}

# --- 5. Parámetros del Algoritmo WMMSE-MM / SCRPA ---
PESOS_USUARIOS = np.array([1.0, 1.0, 1.0])  # <--- [MODIFICABLE] Prioridad de cada usuario en la prueba Multiusuario
TX_POWER_W = 1.0                            # <--- [MODIFICABLE] Potencia de transmisión de referencia (W)
NOISE_POWER_W = 1e-10                       # <--- [MODIFICABLE] Potencia de ruido estimada (W)
MAX_ITER_WMMSE = 150                        # <--- [MODIFICABLE] Número máximo de iteraciones del algoritmo

print(f"✓ Configuración lista: RIS de {RIS_NX}x{RIS_NY} ({RIS_BITS} bits) a {FREQ_HZ/1e9:.1f} GHz.")


# %%
# =============================================================================
# CELDA 3: ESTIMACIÓN DEL CANAL CON LOS MÓDULOS DE surpRISe (TANG GENERAL)
# =============================================================================
# En esta celda se instancian los objetos RIS, Antenna y TangGeneralChannel
# para obtener los vectores de canal TX->RIS (G) y RIS->RX (H_r) en campo cercano.

def estimar_canales_tang_general(tx_pos, usuarios_dict, ris_nx, ris_ny, dx, dy, freq):
    """
    Construye los objetos físicos del repositorio surpRISe y extrae la respuesta
    del canal elemento a elemento en campo cercano (TangGeneralChannel).
    """
    N_total = ris_nx * ris_ny
    k0 = 2 * np.pi / (3e8 / freq)

    # =========================================================================
    # [NOTA]: Si tus clases RIS, Antenna o TangGeneralChannel usan nombres de
    # argumentos distintos en su __init__, ajústalos en las 3 líneas siguientes:
    # =========================================================================
    try:
        ris_obj = RIS(nx=ris_nx, ny=ris_ny, dx=dx, dy=dy)                 # <--- [MODIFICABLE según tu ris.py]
        tx_obj = Antenna(position=tx_pos)                                 # <--- [MODIFICABLE según tu antenna.py]
        usa_objetos_nativos = True
    except TypeError:
        usa_objetos_nativos = False

    # Coordenadas 3D de cada celda de la RIS para evaluar el canal de Tang General
    x_grid = (np.arange(ris_nx) - (ris_nx - 1) / 2.0) * dx
    z_grid = (np.arange(ris_ny) - (ris_ny - 1) / 2.0) * dy
    xx, zz = np.meshgrid(x_grid, z_grid)
    elem_coords = np.column_stack([xx.ravel(), np.zeros(N_total), zz.ravel()])

    # Tramo 1: Vector de canal TX -> cada celda de la RIS (G)
    vec_tx = tx_pos - elem_coords
    d_tx = np.linalg.norm(vec_tx, axis=1)
    cos_tx = np.clip(np.abs(vec_tx[:, 1]) / d_tx, 1e-6, 1.0)
    G = (np.sqrt(dx * dy * cos_tx) / (4 * np.pi * d_tx) * np.exp(-1j * k0 * d_tx))[:, None]

    # Tramo 2: Vectores de canal cada celda de la RIS -> cada Usuario RX (H_r)
    H_r_columnas = []
    for nombre_u, rx_pos in usuarios_dict.items():
        if usa_objetos_nativos:
            rx_obj = Antenna(position=rx_pos)                             # <--- [MODIFICABLE según tu antenna.py]
            ch_obj = TangGeneralChannel(tx=tx_obj, ris=ris_obj, rx=rx_obj, freq=freq) # <--- [MODIFICABLE]
            # Si tu clase TangGeneralChannel expone directamente los vectores TX->RIS y RIS->RX:
            if hasattr(ch_obj, "h_ris2rx") and hasattr(ch_obj, "h_tx2ris"):
                G = np.asarray(ch_obj.h_tx2ris).reshape(-1, 1)
                h_rk = np.asarray(ch_obj.h_ris2rx).reshape(-1)
                H_r_columnas.append(h_rk)
                continue

        # Evaluación directa de la ecuación general de Tang (onda esférica en campo cercano)
        vec_rx = rx_pos - elem_coords
        d_rx = np.linalg.norm(vec_rx, axis=1)
        cos_rx = np.clip(np.abs(vec_rx[:, 1]) / d_rx, 1e-6, 1.0)
        h_rk = (np.sqrt(dx * dy * cos_rx) / (4 * np.pi * d_rx)) * np.exp(-1j * k0 * d_rx)
        H_r_columnas.append(h_rk)

    H_r_matriz = np.column_stack(H_r_columnas)  # Dimensión: (N_elementos, K_usuarios)
    return G, H_r_matriz

G_canal, H_r_canales = estimar_canales_tang_general(
    TX_POS, USUARIOS_RX_POS, RIS_NX, RIS_NY, RIS_DX, RIS_DY, FREQ_HZ
)
print(f"✓ Canales estimados -> G: {G_canal.shape}, H_r: {H_r_canales.shape}")


# %%
# =============================================================================
# CELDA 4: ALGORITMO WMMSE-MM / SCRPA (USUARIO ÚNICO Y MULTIUSUARIO)
# =============================================================================

def cuantizar_fases_ris(fases_rad, n_bits):
    """Convierte fases continuas [0, 2*pi) en fases discretas y estados enteros (0..2^B-1)."""
    fases_mod = np.mod(fases_rad, 2 * np.pi)
    if n_bits is None or n_bits <= 0:
        return fases_mod, np.zeros_like(fases_mod, dtype=int)
    niveles = 2 ** n_bits
    paso_rad = (2 * np.pi) / niveles
    estados_int = np.round(fases_mod / paso_rad).astype(int) % niveles
    fases_cuant_rad = estados_int * paso_rad
    return fases_cuant_rad, estados_int


def optimizar_ris_wmmse_mm(G, H_r, pesos=None, n_bits=1, p_tx=1.0, sigma2=1e-10, max_iter=150):
    """
    Calcula la configuración óptima de la RIS mediante SCRPA (si K=1 usuario)
    o WMMSE-MM (si K>1 usuarios).
    """
    if H_r.ndim == 1:
        H_r = H_r[:, None]
    N, K = H_r.shape

    if pesos is None:
        pesos = np.ones(K)

    # Canal en cascada por cada usuario k: a_k = diag(h_{r,k}^*) * g
    A = H_r.conj() * G[:, 0:1]  # (N, K)

    # --- CASO 1: USUARIO ÚNICO (SCRPA - Alineación de fase cerrada en 1 paso) ---
    if K == 1:
        fases_opt_cont = np.mod(-np.angle(A[:, 0]), 2 * np.pi)
        fases_q_rad, estados_int = cuantizar_fases_ris(fases_opt_cont, n_bits)
        theta_q = np.exp(1j * fases_q_rad)
        h_eff = np.vdot(A[:, 0], theta_q)
        wsr_final = float(pesos[0] * np.log2(1.0 + (p_tx * np.abs(h_eff) ** 2) / sigma2))
        return {
            "fases_rad": fases_q_rad,
            "fases_deg": np.rad2deg(fases_q_rad),
            "estados_digitales": estados_int.reshape((RIS_NX, RIS_NY)),
            "theta": theta_q,
            "historico_wsr": [wsr_final],
        }

    # --- CASO 2: MULTIUSUARIO (WMMSE-MM iterativo) ---
    theta = np.exp(-1j * np.angle(A @ pesos))
    historico_wsr = []

    for _ in range(max_iter):
        # Canal escalar efectivo hacia cada uno de los K usuarios
        h_eff = np.array([np.vdot(A[:, k], theta) for k in range(K)])
        snr = (p_tx * (np.abs(h_eff) ** 2)) / sigma2
        wsr = float(np.sum(pesos * np.log2(1.0 + snr)))
        historico_wsr.append(wsr)

        if len(historico_wsr) > 1 and abs(historico_wsr[-1] - historico_wsr[-2]) < 1e-6:
            break

        # Actualización de receptores MMSE (u_k) y pesos de error (rho_k)
        rx_signal = np.sqrt(p_tx) * h_eff
        u = rx_signal / (np.abs(rx_signal) ** 2 + sigma2)
        mse = np.maximum(1.0 - np.real(u.conj() * rx_signal), 1e-12)
        rho = pesos / mse

        # Construcción de los términos cuadrático (U) y lineal (v) del mayorizador MM
        U = np.zeros((N, N), dtype=complex)
        v = np.zeros(N, dtype=complex)
        for k in range(K):
            a_k = A[:, k].conj()
            U += rho[k] * (np.abs(u[k]) ** 2) * p_tx * np.outer(a_k, a_k.conj())
            v += rho[k] * (u[k].conj() * np.sqrt(p_tx)) * a_k

        U = 0.5 * (U + U.conj().T)
        lambda_max = float(np.max(np.linalg.eigvalsh(U)))

        # Iteraciones internas de Majorization-Minimization (MM)
        for _ in range(25):
            z = (lambda_max * theta - U @ theta) + v
            theta = np.exp(1j * np.angle(z))

    fases_opt_cont = np.mod(np.angle(theta), 2 * np.pi)
    fases_q_rad, estados_int = cuantizar_fases_ris(fases_opt_cont, n_bits)

    return {
        "fases_rad": fases_q_rad,
        "fases_deg": np.rad2deg(fases_q_rad),
        "estados_digitales": estados_int.reshape((RIS_NX, RIS_NY)),
        "theta": np.exp(1j * fases_q_rad),
        "historico_wsr": historico_wsr,
    }

print("✓ Funciones de optimización WMMSE-MM / SCRPA cargadas.")


# %%
# =============================================================================
# CELDA 5: CONEXIÓN Y ENVÍO AL SERVIDOR DE LA RIS (ris_hardware_server.py)
# =============================================================================

def enviar_configuracion_a_ris(matriz_estados, nombre_prueba="test"):
    """
    Envía la matriz 2D de estados digitales a ris_hardware_server.py y guarda
    una copia local en CSV por seguridad.
    """
    # 1. Guardamos copia local por si deseas inspeccionarla
    nombre_csv = f"config_ris_{nombre_prueba}.csv"
    np.savetxt(nombre_csv, matriz_estados, fmt="%d", delimiter=",")
    print(f"   [Archivo local guardado]: {nombre_csv}")

    if not ENVIAR_AL_SERVIDOR:
        print("   [Modo Simulación]: Envío al servidor desactivado (ENVIAR_AL_SERVIDOR = False).")
        return

    # 2. Formato JSON que se envía al servidor (ajústalo si tu servidor espera otra clave)
    payload = {
        "pattern": matriz_estados.tolist(),   # <--- [MODIFICABLE] Clave JSON que espera ris_hardware_server.py
        "name": nombre_prueba
    }
    datos_bytes = json.dumps(payload).encode("utf-8")

    try:
        if SERVER_PROTOCOL.lower() == "http":
            url = f"http://{SERVER_IP}:{SERVER_PORT}{SERVER_ENDPOINT}"
            req = urllib.request.Request(
                url, data=datos_bytes, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                print(f"   ✓ [RIS SERVER HTTP] Configurado con éxito. Respuesta: {response.read().decode()}")
        else:
            # Envío directo por Socket TCP
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((SERVER_IP, SERVER_PORT))
                s.sendall(datos_bytes)
                respuesta = s.recv(1024).decode("utf-8", errors="ignore")
                print(f"   ✓ [RIS SERVER TCP] Configurado con éxito. Respuesta: {respuesta}")
    except Exception as e:
        print(f"   ⚠ [AVISO CONEXIÓN]: No se pudo contactar con {SERVER_IP}:{SERVER_PORT} ({e}).")
        print(f"     Puedes cargar manualmente el archivo '{nombre_csv}' en el servidor.")


# %%
# =============================================================================
# CELDA 6: PRUEBA 1 - CONFIGURACIÓN DE LA RIS PARA USUARIOS ÚNICOS (UNO A UNO)
# =============================================================================
# Ejecuta esta celda para apuntar la RIS a cada usuario individualmente.
# En cada iteración se configura la RIS y el programa espera a que midas con el VNA.

resultados_individuales = {}
nombres_usuarios = list(USUARIOS_RX_POS.keys())

print("=================================================================")
print(" INICIANDO PRUEBA 1: USUARIOS ÚNICOS (SINGLE-USER)")
print("=================================================================")

for idx, nombre_usuario in enumerate(nombres_usuarios):
    print(f"\n---> Calculando RIS para apuntar exclusivamente a: {nombre_usuario}")
    
    # Extraemos solo la columna del canal correspondiente a este usuario
    h_r_individual = H_r_canales[:, idx]
    
    # Ejecutamos optimización para 1 usuario
    res_su = optimizar_ris_wmmse_mm(
        G=G_canal,
        H_r=h_r_individual,
        n_bits=RIS_BITS,
        p_tx=TX_POWER_W,
        sigma2=NOISE_POWER_W
    )
    resultados_individuales[nombre_usuario] = res_su
    
    # Enviamos al servidor de la RIS para actualizar la placa física
    enviar_configuracion_a_ris(res_su["estados_digitales"], nombre_prueba=f"SU_{nombre_usuario}")
    
    # Pausa interactiva para que puedas medir con el VNA antes de cambiar al siguiente usuario
    input(f"   >>> RIS apuntando a {nombre_usuario}. Realiza tu medida en el VNA y pulsa [ENTER] para continuar...")


# %%
# =============================================================================
# CELDA 7: PRUEBA 2 - CONFIGURACIÓN DE LA RIS PARA MULTIUSUARIO (TODOS A LA VEZ)
# =============================================================================
# Ejecuta esta celda cuando quieras que la RIS configure un patrón multihaz
# en campo cercano para dar servicio simultáneo a todos los usuarios.

print("\n=================================================================")
print(" INICIANDO PRUEBA 2: MULTIUSUARIO CONJUNTO (WMMSE-MM)")
print("=================================================================")

res_multiusuario = optimizar_ris_wmmse_mm(
    G=G_canal,
    H_r=H_r_canales,
    pesos=PESOS_USUARIOS,
    n_bits=RIS_BITS,
    p_tx=TX_POWER_W,
    sigma2=NOISE_POWER_W,
    max_iter=MAX_ITER_WMMSE
)

# Enviamos el patrón multiusuario al servidor de la RIS
enviar_configuracion_a_ris(res_multiusuario["estados_digitales"], nombre_prueba="MU_Todos")

print("   >>> RIS configurada en modo MULTIUSUARIO. Ya puedes medir con el VNA en todos los puntos.")


# %%
# =============================================================================
# CELDA 8: VISUALIZACIÓN DE LOS PATRONES DE FASE Y CONVERGENCIA WMMSE-MM
# =============================================================================
num_plots = len(nombres_usuarios) + 1
fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))

# 1. Mapas de fase de cada usuario único
for idx, nombre_usuario in enumerate(nombres_usuarios):
    mapa_grados = resultados_individuales[nombre_usuario]["fases_deg"].reshape((RIS_NX, RIS_NY))
    im = axes[idx].imshow(mapa_grados, cmap="twilight", vmin=0, vmax=360)
    axes[idx].set_title(f"SU: {nombre_usuario}")
    axes[idx].set_xlabel("Columna X")
    axes[idx].set_ylabel("Fila Y")

# 2. Mapa de fase del modo Multiusuario
mapa_mu_grados = res_multiusuario["fases_deg"].reshape((RIS_NX, RIS_NY))
im = axes[-1].imshow(mapa_mu_grados, cmap="twilight", vmin=0, vmax=360)
axes[-1].set_title("MU: Todos (WMMSE-MM)")
axes[-1].set_xlabel("Columna X")

fig.colorbar(im, ax=axes.ravel().tolist(), label="Fase (Grados)", shrink=0.8)
plt.suptitle(f"Configuraciones de la RIS en Campo Cercano ({RIS_BITS} bits)", fontsize=13)
plt.show()

# 3. Curva de convergencia del algoritmo WMMSE-MM Multiusuario
plt.figure(figsize=(6, 4))
plt.plot(res_multiusuario["historico_wsr"], marker="o", markersize=3)
plt.title("Convergencia del Algoritmo WMMSE-MM (Multiusuario)")
plt.xlabel("Iteración")
plt.ylabel("Weighted Sum-Rate Teórico (bps/Hz)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()