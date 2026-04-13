"""
kedm_engine.py
==============
Core KEDM solver engine.
Based on: Osanyinpeju, K. L. (2025). JSID 7(1). DOI: 10.20372/jsid/2025-340
Extended to n variables.
"""

from fractions import Fraction
import time


def det2(a, b, c, d):
    """
    The ONLY arithmetic operation in KEDM.
    Computes the 2x2 determinant:
        | a  b |
        | c  d |  =  a*d - b*c
    """
    return a * d - b * c


def order_rows(A, b, elim_col):
    """Interleave rows so no consecutive pair is both zero at elim_col."""
    m = len(b)
    nonzero = [r for r in range(m) if A[r][elim_col] != 0]
    zeros   = [r for r in range(m) if A[r][elim_col] == 0]
    ordered = []
    nz, z = 0, 0
    for _ in range(m):
        if nz < len(nonzero):
            ordered.append(nonzero[nz]); nz += 1
        elif z < len(zeros):
            ordered.append(zeros[z]);   z  += 1
    return [A[r] for r in ordered], [b[r] for r in ordered]


def kedm_eliminate(A, b, elim_col, steps_log):
    """
    One KEDM reduction step: eliminates variable at elim_col.
    Returns reduced system and appends step details to steps_log.
    """
    m = len(b)
    cols = [j for j in range(len(A[0])) if j != elim_col]
    A_new, b_new = [], []
    det2_calls = 0

    for i in range(m - 1):
        e1, e2 = A[i][elim_col], A[i + 1][elim_col]
        new_row = []
        for old_j in cols:
            val = det2(A[i][old_j], e1, A[i + 1][old_j], e2)
            new_row.append(val)
            det2_calls += 1
        rhs_val = det2(b[i], e1, b[i + 1], e2)
        b_new.append(rhs_val)
        det2_calls += 1
        A_new.append(new_row)

        steps_log.append({
            "type": "det2_pair",
            "row1": i,
            "row2": i + 1,
            "elim_col": elim_col,
            "e1": str(e1),
            "e2": str(e2),
            "rhs": str(rhs_val),
        })

    return A_new, b_new, cols, det2_calls


def kedm_solve_one(A_input, b_input, target_var, var_names=None):
    """
    Solve for one variable using KEDM with full step logging.
    Returns (value, steps_log, det2_count).
    """
    n = len(b_input)
    A = [[Fraction(x) for x in row] for row in A_input]
    b = [Fraction(x) for x in b_input]
    current_cols = list(range(n))
    steps_log = []
    total_det2 = 0

    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]

    steps_log.append({
        "type": "start",
        "message": f"Solving for {var_names[target_var]} using KEDM",
        "system_size": n,
    })

    for elim_var in range(n):
        if elim_var == target_var or elim_var not in current_cols:
            continue

        elim_col = current_cols.index(elim_var)
        A, b = order_rows(A, b, elim_col)

        steps_log.append({
            "type": "elimination_start",
            "eliminating": var_names[elim_var] if elim_var < len(var_names) else f"x{elim_var+1}",
            "remaining_size": len(b),
        })

        A, b, remaining, det2_count = kedm_eliminate(A, b, elim_col, steps_log)
        current_cols = [current_cols[i] for i in remaining]
        total_det2 += det2_count

        steps_log.append({
            "type": "elimination_done",
            "new_size": len(b),
            "det2_used": det2_count,
        })

    if not A or A[0][0] == 0:
        steps_log.append({"type": "error", "message": "Singular system - no unique solution"})
        return None, steps_log, total_det2

    result = b[0] / A[0][0]
    steps_log.append({
        "type": "result",
        "variable": var_names[target_var],
        "value": str(result),
        "float_value": float(result),
    })

    return float(result), steps_log, total_det2


def kedm_solve_system(A, b, var_names=None):
    """
    Solve full n-variable system. Returns full solve report.
    """
    n = len(b)
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]

    t_start = time.perf_counter()
    solutions = {}
    all_steps = []
    total_det2 = 0

    for var_idx in range(n):
        val, steps, d2 = kedm_solve_one(A, b, var_idx, var_names)
        solutions[var_names[var_idx]] = val
        all_steps.extend(steps)
        total_det2 += d2

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    return {
        "solutions": solutions,
        "steps": all_steps,
        "total_det2_calls": total_det2,
        "elapsed_ms": round(elapsed_ms, 4),
        "n_variables": n,
        "method": "KEDM",
    }
