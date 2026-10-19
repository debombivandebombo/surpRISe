import numpy as np
from surpRISe.optimization import WMMSE_MM_Optimizer


def generate_los_channel(num_elements: int, num_antennas: int, angle_deg: float) -> np.ndarray:
    """Generador auxiliar de canal en línea de vista (ULA/UPA simplificado) para pruebas."""
    idx = np.arange(num_elements)
    steering = np.exp(1j * np.pi * idx * np.sin(np.deg2rad(angle_deg)))
    if num_antennas == 1:
        return steering[:, None] / np.sqrt(num_elements)
    tx_idx = np.arange(num_antennas)
    tx_steering = np.exp(-1j * np.pi * tx_idx * np.sin(np.deg2rad(15.0)))
    return np.outer(steering, tx_steering) / np.sqrt(num_elements)


def test_single_user():
    print("=== PRUEBA 1: USUARIO ÚNICO (Apuntado de la RIS) ===")
    N_ris = 64  # Ej. RIS de 8x8 elementos
    M_tx = 1    # Antena bocina (Horn) en TX

    # Nota: Aquí puedes sustituir G y h_r por la salida de tu clase en
    # src/surpRISe/channels/tx2ris2rx/tang_ff_channel.py
    G = generate_los_channel(N_ris, M_tx, angle_deg=-20.0)
    h_r = generate_los_channel(N_ris, 1, angle_deg=35.0)[:, 0]

    optimizer = WMMSE_MM_Optimizer(
        num_ris_elements=N_ris,
        num_tx_antennas=M_tx,
        tx_power_max=1.0,
        noise_power=1e-3,
        ris_bits=2,  # Prueba con cuantización de 2 bits (0°, 90°, 180°, 270°)
    )

    res = optimizer.optimize(G=G, H_r=h_r, method="wmmse_mm")
    print(f"Tasa inicial (sin optimizar): {res['wsr_history'][0]:.2f} bps/Hz")
    print(f"Tasa final (con WMMSE-MM):    {res['wsr_history'][-1]:.2f} bps/Hz")
    print(f"Primeras 8 fases cuantizadas (grados): {res['phases_deg'][:8]}")
    return res["phases_deg"].reshape(8, 8)


def test_multi_user():
    print("\n=== PRUEBA 2: MULTIUSUARIO (K=3 Usuarios) ===")
    N_ris = 64
    M_tx = 4
    K_users = 3

    G = generate_los_channel(N_ris, M_tx, angle_deg=-15.0)
    angles_rx = [-30.0, 10.0, 45.0]
    H_r = np.column_stack([
        generate_los_channel(N_ris, 1, angle_deg=ang)[:, 0] for ang in angles_rx
    ])

    optimizer = WMMSE_MM_Optimizer(
        num_ris_elements=N_ris,
        num_tx_antennas=M_tx,
        tx_power_max=1.0,
        noise_power=1e-2,
        ris_bits=2,
    )

    res = optimizer.optimize(G=G, H_r=H_r, method="wmmse_mm")
    print(f"Sum-Rate inicial (K={K_users}): {res['wsr_history'][0]:.2f} bps/Hz")
    print(f"Sum-Rate final   (K={K_users}): {res['wsr_history'][-1]:.2f} bps/Hz")
    print(f"Iteraciones hasta converger: {len(res['wsr_history']) - 1}")


if __name__ == "__main__":
    phase_matrix_8x8 = test_single_user()
    test_multi_user()