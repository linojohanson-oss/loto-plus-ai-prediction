# pipeline_loto_plus.py

import os
import sys
import time
import subprocess
from datetime import datetime


# =========================
# CONFIG
# =========================
SCRAPER_SCRIPT = "loto_plus_incremental_a_excel.py"
ANALISIS_SCRIPT = "loto_plus_analisis_estadistico.py"
PRONOSTICO_SCRIPT = "loto_plus_pronosticos.py"
DL_SCRIPT = "ia_loto_plus_dl.py"

MODALIDADES = [
    "Tradicional",
    "Match",
    "Desquite",
    "Sale o Sale",
]

DEFAULT_N_SORTEOS = 500


# =========================
# UTILS
# =========================
def banner(txt):
    print("\n" + "=" * 80)
    print(txt)
    print("=" * 80)


def run_command(cmd, step_name):
    banner(f"▶ {step_name}")

    print("COMANDO:")
    print(" ".join(cmd))
    print()

    start = time.time()

    result = subprocess.run(cmd)

    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"❌ Error en: {step_name} | returncode={result.returncode}"
        )

    print(f"\n✅ OK: {step_name}")
    print(f"⏱ Tiempo: {elapsed:.2f} segundos")


# =========================
# VALIDACIONES
# =========================
def validate_scripts():
    required = [
        SCRAPER_SCRIPT,
        ANALISIS_SCRIPT,
        PRONOSTICO_SCRIPT,
        DL_SCRIPT,
    ]

    missing = [f for f in required if not os.path.exists(f)]

    if missing:
        raise FileNotFoundError(
            "Faltan scripts requeridos:\n" + "\n".join(missing)
        )


# =========================
# MAIN
# =========================
def main():
    validate_scripts()

    n_sorteos = (
        int(sys.argv[1])
        if len(sys.argv) >= 2
        else DEFAULT_N_SORTEOS
    )

    start_total = time.time()

    banner("PIPELINE MAESTRO LOTO PLUS")

    print(f"Fecha inicio: {datetime.now()}")
    print(f"Cantidad sorteos objetivo: {n_sorteos}")

    # ======================================================
    # 1. SCRAPER
    # ======================================================
    run_command(
        [
            sys.executable,
            SCRAPER_SCRIPT,
            str(n_sorteos),
            "loto_plus_incremental.xlsx",
        ],
        "SCRAPER HISTÓRICO"
    )

    # ======================================================
    # 2. ANÁLISIS ESTADÍSTICO
    # ======================================================
    for modalidad in MODALIDADES:
        run_command(
            [
                sys.executable,
                ANALISIS_SCRIPT,
                modalidad,
                "200",
            ],
            f"ANÁLISIS ESTADÍSTICO - {modalidad}"
        )

    # ======================================================
    # 3. PRONÓSTICOS HÍBRIDOS
    # ======================================================
    run_command(
        [
            sys.executable,
            PRONOSTICO_SCRIPT,
            "--in",
            "loto_plus_incremental.xlsx",
            "--out",
            "pronosticos.xlsx",
            "--recent",
            "200",
            "--preds",
            "12",
            "--use_transformer",
            "--epochs",
            "35",
            "--seq",
            "30",
            "--alpha",
            "0.55",
        ],
        "PRONÓSTICOS HÍBRIDOS"
    )

    # ======================================================
    # 4. DEEP LEARNING
    # ======================================================
    for modalidad in MODALIDADES:
        run_command(
            [
                sys.executable,
                DL_SCRIPT,
                modalidad,
            ],
            f"DEEP LEARNING - {modalidad}"
        )

    # ======================================================
    # FINAL
    # ======================================================
    elapsed_total = time.time() - start_total

    banner("PIPELINE COMPLETADO")

    print("\n✅ TODO FINALIZADO")
    print(f"⏱ Tiempo total: {elapsed_total / 60:.2f} minutos")
    print(f"📅 Finalizado: {datetime.now()}")

    print("\nARCHIVOS GENERADOS:")
    print("- loto_plus_incremental.xlsx")
    print("- analisis_estadistico_*.xlsx")
    print("- pronosticos.xlsx")
    print("- prediccion_dl_loto_plus.xlsx")


if __name__ == "__main__":
    main()