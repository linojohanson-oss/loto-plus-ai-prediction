# optimizador_cobertura_loto.py
# ------------------------------------------------------------
# OPTIMIZADOR DE COBERTURA — LOTO PLUS
#
# Objetivo:
# generar 4 o 5 tickets optimizados:
#
# - mínimo solapamiento
# - máxima diversidad
# - buena cobertura
# - balance estadístico razonable
#
# NO intenta predecir.
# Optimiza estructura del portafolio.
#
# Exporta:
#   optimizador_cobertura.xlsx
#
# Uso:
#   python optimizador_cobertura_loto.py
#
# Opcional:
#   python optimizador_cobertura_loto.py 5
# ------------------------------------------------------------

import sys
import random
import itertools
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
MAX_NUM = 45

NUMBERS_PER_TICKET = 6

DEFAULT_TICKETS = 5

N_CANDIDATES = 4000

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# =========================
# HELPERS
# =========================
def generate_ticket():

    nums = random.sample(
        range(1, MAX_NUM + 1),
        NUMBERS_PER_TICKET
    )

    return sorted(nums)


def ticket_sum(ticket):

    return sum(ticket)


def odd_even_balance(ticket):

    odds = sum(n % 2 for n in ticket)

    evens = 6 - odds

    return odds, evens


def low_high_balance(ticket):

    low = sum(n <= 22 for n in ticket)

    high = 6 - low

    return low, high


def has_long_sequence(ticket):

    seq = 1

    for i in range(1, len(ticket)):

        if ticket[i] == ticket[i - 1] + 1:

            seq += 1

            if seq >= 4:
                return True

        else:
            seq = 1

    return False


def valid_ticket(ticket):

    # evitar secuencias largas
    if has_long_sequence(ticket):
        return False

    # balance par/impar
    odds, evens = odd_even_balance(ticket)

    if odds == 0 or evens == 0:
        return False

    # balance bajo/alto
    low, high = low_high_balance(ticket)

    if low == 0 or high == 0:
        return False

    # suma razonable
    s = ticket_sum(ticket)

    if s < 70 or s > 210:
        return False

    return True


# =========================
# OVERLAP
# =========================
def overlap(a, b):

    return len(
        set(a).intersection(set(b))
    )


# =========================
# SCORE PORTFOLIO
# =========================
def portfolio_score(tickets):

    # =====================
    # cobertura total
    # =====================
    all_nums = set()

    for t in tickets:
        all_nums.update(t)

    coverage = len(all_nums)

    # =====================
    # overlap penalty
    # =====================
    overlap_penalty = 0

    for a, b in itertools.combinations(tickets, 2):

        ov = overlap(a, b)

        overlap_penalty += ov ** 2

    # =====================
    # diversity bonus
    # =====================
    sums = [ticket_sum(t) for t in tickets]

    diversity = np.std(sums)

    # =====================
    # final score
    # =====================
    score = (
        coverage * 10
        - overlap_penalty * 4
        + diversity
    )

    return score


# =========================
# BUILD PORTFOLIO
# =========================
def generate_portfolio(n_tickets):

    tickets = []

    tries = 0

    while len(tickets) < n_tickets:

        tries += 1

        if tries > 100000:
            break

        t = generate_ticket()

        if not valid_ticket(t):
            continue

        # evitar tickets duplicados
        if t in tickets:
            continue

        # limitar overlap fuerte
        bad = False

        for existing in tickets:

            if overlap(t, existing) >= 4:
                bad = True
                break

        if bad:
            continue

        tickets.append(t)

    return tickets


# =========================
# OPTIMIZATION
# =========================
def optimize(n_tickets):

    best_score = -1e9

    best_portfolio = None

    for i in range(N_CANDIDATES):

        portfolio = generate_portfolio(
            n_tickets
        )

        if len(portfolio) != n_tickets:
            continue

        score = portfolio_score(portfolio)

        if score > best_score:

            best_score = score

            best_portfolio = portfolio

        if (i + 1) % 500 == 0:

            print(
                f"Iteración {i + 1}/{N_CANDIDATES} | "
                f"Best score={best_score:.2f}"
            )

    return best_portfolio, best_score


# =========================
# EXPORT
# =========================
def export_excel(
    portfolio,
    score
):

    rows = []

    for i, t in enumerate(portfolio, 1):

        rows.append({
            "ticket": i,
            "numeros": "-".join(
                f"{x:02d}" for x in t
            ),
            "suma": sum(t),
            "pares": sum(n % 2 == 0 for n in t),
            "impares": sum(n % 2 == 1 for n in t),
        })

    df = pd.DataFrame(rows)

    # cobertura total
    all_nums = set()

    for t in portfolio:
        all_nums.update(t)

    coverage = len(all_nums)

    summary = pd.DataFrame([
        {
            "metric": "score",
            "value": round(score, 4)
        },
        {
            "metric": "coverage_total",
            "value": coverage
        },
    ])

    with pd.ExcelWriter(
        "optimizador_cobertura.xlsx",
        engine="openpyxl"
    ) as wr:

        df.to_excel(
            wr,
            sheet_name="Tickets",
            index=False
        )

        summary.to_excel(
            wr,
            sheet_name="Resumen",
            index=False
        )

    print("\nOK ✅ PORTAFOLIO OPTIMIZADO")
    print("Archivo: optimizador_cobertura.xlsx")


# =========================
# MAIN
# =========================
def main():

    n_tickets = (
        int(sys.argv[1])
        if len(sys.argv) >= 2
        else DEFAULT_TICKETS
    )

    print("\n==============================")
    print("OPTIMIZADOR DE COBERTURA")
    print("==============================")

    print(f"Tickets: {n_tickets}")

    portfolio, score = optimize(n_tickets)

    print("\nMEJOR PORTAFOLIO:\n")

    for i, t in enumerate(portfolio, 1):

        print(
            f"{i}. "
            + " ".join(f"{x:02d}" for x in t)
        )

    print(f"\nScore: {score:.2f}")

    export_excel(
        portfolio,
        score
    )


if __name__ == "__main__":
    main()