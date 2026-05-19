import sys
from itertools import combinations
from typing import List, Tuple, Dict, Optional

import pandas as pd


# =========================
# CONFIG
# =========================
INPUT_XLSX = "loto_plus_incremental.xlsx"
OUTPUT_XLSX = "analisis_estadistico_loto_plus.xlsx"

MAX_NUM = 45
NUM_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

VALID_SHEETS = ["Tradicional", "Match", "Desquite", "Sale o Sale"]


# =========================
# HELPERS
# =========================
def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_draws(sheet_name: str, last_n: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_excel(INPUT_XLSX, sheet_name=sheet_name)

    # Normalizar columnas esperadas
    if "sorteo" not in df.columns:
        raise ValueError(f"La hoja '{sheet_name}' no tiene columna 'sorteo'.")

    df["sorteo"] = pd.to_numeric(df["sorteo"], errors="coerce").fillna(0).astype(int)

    for c in NUM_COLS:
        if c not in df.columns:
            raise ValueError(f"La hoja '{sheet_name}' no tiene la columna '{c}'.")
        df[c] = _coerce_int_series(df[c])

    # Orden: más reciente primero
    df = df.sort_values("sorteo", ascending=False).reset_index(drop=True)

    # Recorte últimos N sorteos
    if last_n is not None and last_n > 0:
        df = df.head(last_n).copy()

    # Quitar filas con menos de 6 números válidos (por seguridad)
    valid_count = df[NUM_COLS].notna().sum(axis=1)
    df = df[valid_count >= 6].reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"No hay datos válidos para analizar en '{sheet_name}'.")

    return df


def flatten_numbers(df: pd.DataFrame) -> List[int]:
    vals = df[NUM_COLS].values.flatten()
    out = []
    for v in vals:
        if pd.isna(v):
            continue
        iv = int(v)
        if 1 <= iv <= MAX_NUM:
            out.append(iv)
    return out


def compute_frequency(df: pd.DataFrame) -> pd.DataFrame:
    all_nums = flatten_numbers(df)
    freq = pd.Series(all_nums).value_counts().sort_index()

    base = pd.DataFrame({"numero": range(1, MAX_NUM + 1)})
    base["frecuencia"] = base["numero"].map(lambda x: int(freq.get(x, 0)))
    total = int(base["frecuencia"].sum())
    base["frecuencia_pct"] = base["frecuencia"].map(lambda x: (x / total) if total else 0.0)

    return base


def compute_delay(df: pd.DataFrame) -> pd.Series:
    # atraso: cuántos sorteos hacia atrás hasta la última aparición (0 = salió en el último)
    atrasos: Dict[int, Optional[int]] = {}

    # armamos una lista por fila (más reciente primero) para acelerar
    rows = df[NUM_COLS].astype(int).values.tolist()

    for num in range(1, MAX_NUM + 1):
        delay = None
        for i, nums in enumerate(rows):
            if num in nums:
                delay = i
                break
        atrasos[num] = delay

    return pd.Series(atrasos, name="atraso")


def compute_pairs(df: pd.DataFrame) -> pd.DataFrame:
    counter: Dict[Tuple[int, int], int] = {}
    rows = df[NUM_COLS].astype(int).values.tolist()

    for nums in rows:
        nums = sorted(nums)
        for a, b in combinations(nums, 2):
            counter[(a, b)] = counter.get((a, b), 0) + 1

    pairs_df = pd.DataFrame(
        [(a, b, c) for (a, b), c in counter.items()],
        columns=["n1", "n2", "frecuencia"]
    ).sort_values("frecuencia", ascending=False).reset_index(drop=True)

    return pairs_df


def compute_triples(df: pd.DataFrame, top_limit: int = 300) -> pd.DataFrame:
    # triples pueden ser muchos. Dejamos top_limit para no explotar Excel
    counter: Dict[Tuple[int, int, int], int] = {}
    rows = df[NUM_COLS].astype(int).values.tolist()

    for nums in rows:
        nums = sorted(nums)
        for a, b, c in combinations(nums, 3):
            counter[(a, b, c)] = counter.get((a, b, c), 0) + 1

    triples_df = pd.DataFrame(
        [(a, b, c, f) for (a, b, c), f in counter.items()],
        columns=["n1", "n2", "n3", "frecuencia"]
    ).sort_values("frecuencia", ascending=False).reset_index(drop=True)

    if top_limit and len(triples_df) > top_limit:
        triples_df = triples_df.head(top_limit).copy()

    return triples_df


def compute_distributions(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[NUM_COLS].astype(int)

    def row_stats(nums: List[int]) -> Dict[str, int]:
        pares = sum(1 for n in nums if n % 2 == 0)
        impares = 6 - pares
        bajos = sum(1 for n in nums if n <= 22)
        altos = 6 - bajos
        suma = sum(nums)
        return {"pares": pares, "impares": impares, "bajos_1_22": bajos, "altos_23_45": altos, "suma": suma}

    stats = [row_stats(r) for r in rows.values.tolist()]
    s = pd.DataFrame(stats)

    # Resumen (promedios y distribución simple)
    resumen = pd.DataFrame({
        "metric": ["prom_pares", "prom_impares", "prom_bajos_1_22", "prom_altos_23_45", "prom_suma"],
        "valor": [
            s["pares"].mean(),
            s["impares"].mean(),
            s["bajos_1_22"].mean(),
            s["altos_23_45"].mean(),
            s["suma"].mean(),
        ]
    })

    # Histograma simple de suma
    bins = [0, 90, 105, 120, 135, 150, 165, 180, 999]
    labels = ["<=90", "91-105", "106-120", "121-135", "136-150", "151-165", "166-180", ">=181"]
    s["rango_suma"] = pd.cut(s["suma"], bins=bins, labels=labels, include_lowest=True)
    hist_suma = s["rango_suma"].value_counts().reindex(labels, fill_value=0).reset_index()
    hist_suma.columns = ["rango_suma", "conteo"]

    return resumen, hist_suma


def build_ranking(freq_df: pd.DataFrame, delay_s: pd.Series) -> pd.DataFrame:
    df = freq_df.copy()
    df["atraso"] = df["numero"].map(delay_s.to_dict())

    # Score combinado:
    # - normalizamos frecuencia y atraso (a mayor atraso, más “deuda”)
    max_f = df["frecuencia"].max() if df["frecuencia"].max() else 1
    max_a = df["atraso"].dropna().max() if df["atraso"].dropna().max() is not None else 1

    df["freq_norm"] = df["frecuencia"] / max_f
    df["atraso_norm"] = df["atraso"].fillna(max_a) / (max_a if max_a else 1)

    # pesos (podés ajustar): frecuencia manda, atraso acompaña
    w_freq = 0.70
    w_atraso = 0.30
    df["score"] = w_freq * df["freq_norm"] + w_atraso * df["atraso_norm"]

    # Orden principal por score, luego por frecuencia y atraso
    df = df.sort_values(["score", "frecuencia", "atraso"], ascending=[False, False, False]).reset_index(drop=True)

    return df


def export_all(
    sheet_name: str,
    last_n: Optional[int],
    ranking_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    triples_df: pd.DataFrame,
    resumen_dist: pd.DataFrame,
    hist_suma: pd.DataFrame,
) -> None:
    suffix = f"{sheet_name}_{last_n or 'ALL'}"
    out_path = OUTPUT_XLSX.replace(".xlsx", f"_{suffix}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        ranking_df.to_excel(writer, sheet_name="Numeros_Ranking", index=False)

        # Top cortes útiles
        ranking_df.head(15).to_excel(writer, sheet_name="Top15_Score", index=False)
        ranking_df.sort_values("frecuencia", ascending=False).head(15).to_excel(writer, sheet_name="Top15_Frecuencia", index=False)

        # Atrasados (los que más tardaron)
        atrasados = ranking_df.dropna(subset=["atraso"]).sort_values("atraso", ascending=False).head(15)
        atrasados.to_excel(writer, sheet_name="Top15_Atraso", index=False)

        pairs_df.head(200).to_excel(writer, sheet_name="Pares_Top200", index=False)
        triples_df.to_excel(writer, sheet_name="Trios_Top", index=False)

        resumen_dist.to_excel(writer, sheet_name="Distrib_Resumen", index=False)
        hist_suma.to_excel(writer, sheet_name="Distrib_Suma", index=False)

    print("OK ✅ Análisis generado")
    print(f"Entrada: {INPUT_XLSX} | Hoja: {sheet_name} | Ventana: {last_n or 'ALL'} sorteos")
    print(f"Salida: {out_path}")


# =========================
# MAIN (CLI)
# =========================
def main():
    # Uso:
    #   python analisis_loto_plus.py Tradicional 200
    #   python analisis_loto_plus.py Desquite 100
    #   python analisis_loto_plus.py Tradicional   (toma ALL)
    sheet = sys.argv[1] if len(sys.argv) >= 2 else "Tradicional"
    last_n = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    if sheet not in VALID_SHEETS:
        raise ValueError(f"Hoja inválida. Usá una de: {VALID_SHEETS}")

    df = load_draws(sheet, last_n=last_n)
    freq_df = compute_frequency(df)
    delay_s = compute_delay(df)
    ranking_df = build_ranking(freq_df, delay_s)

    pairs_df = compute_pairs(df)
    triples_df = compute_triples(df, top_limit=300)

    resumen_dist, hist_suma = compute_distributions(df)

    export_all(
        sheet_name=sheet,
        last_n=last_n,
        ranking_df=ranking_df,
        pairs_df=pairs_df,
        triples_df=triples_df,
        resumen_dist=resumen_dist,
        hist_suma=hist_suma,
    )


if __name__ == "__main__":
    main()
