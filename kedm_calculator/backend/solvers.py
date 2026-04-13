"""
solvers.py
==========
All supported solvers:
  - KEDM (foundation method)
  - Gaussian Elimination
  - Jacobi Iteration
  - Gauss-Seidel Iteration
  - Cramer's Rule
  - Substitution (2-var)
  - Quadratic Formula
  - Arithmetic evaluator
"""

import time
import math
import re
import numpy as np
from fractions import Fraction


# ════════════════════════════════════════════
#  SHARED PARSER
# ════════════════════════════════════════════

def parse_linear_system(text: str):
    """
    Parse a system of linear equations from plain text.
    One equation per line. Supports any variable names.

    Examples:
        2x + 3y = 7
        -4x + y - 2z = 1
        5I1 - 3I2 + I3 = 10
        a - 2b + 4c - d = 5
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        raise ValueError("No equations found.")

    all_vars = []
    var_pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9_]*')

    for line in lines:
        if '=' not in line:
            raise ValueError(f"Missing '=' in: {line}")
        lhs = line.split('=')[0]
        for v in var_pattern.findall(lhs):
            if v not in all_vars:
                all_vars.append(v)

    n = len(all_vars)
    if len(lines) < n:
        raise ValueError(f"Need {n} equations for {n} variables, got {len(lines)}")
    lines = lines[:n]

    A, b = [], []
    for line in lines:
        lhs, rhs = line.split('=', 1)
        rhs_val = float(eval(rhs.strip(), {"__builtins__": {}}))
        coeffs = {v: 0.0 for v in all_vars}

        # Tokenize LHS into terms
        lhs_norm = re.sub(r'(?<=[a-zA-Z0-9])\s*\+', ' +', lhs)
        lhs_norm = re.sub(r'(?<=[a-zA-Z0-9])\s*-', ' -', lhs_norm)
        lhs_norm = lhs_norm.strip()

        term_re = re.compile(r'([+-]?\s*\d*\.?\d*)\s*([a-zA-Z][a-zA-Z0-9_]*)')
        for coeff_str, var in term_re.findall(lhs_norm):
            cs = coeff_str.replace(' ', '')
            if cs in ('', '+'):    coeff = 1.0
            elif cs == '-':        coeff = -1.0
            else:                  coeff = float(cs)
            if var in coeffs:
                coeffs[var] += coeff

        A.append([coeffs[v] for v in all_vars])
        b.append(rhs_val)

    return A, b, all_vars


# ════════════════════════════════════════════
#  KEDM SOLVER  (foundation method)
# ════════════════════════════════════════════

def det2(a, b, c, d):
    return a * d - b * c

def _kedm_order_rows(A, b, elim_col):
    m = len(b)
    nz = [r for r in range(m) if A[r][elim_col] != 0]
    z  = [r for r in range(m) if A[r][elim_col] == 0]
    order = []
    i, j = 0, 0
    for _ in range(m):
        if i < len(nz): order.append(nz[i]); i += 1
        elif j < len(z): order.append(z[j]);  j += 1
    return [A[r] for r in order], [b[r] for r in order]

def _kedm_eliminate(A, b, elim_col, log):
    m = len(b)
    cols = [j for j in range(len(A[0])) if j != elim_col]
    A_new, b_new = [], []
    for i in range(m - 1):
        e1, e2 = A[i][elim_col], A[i+1][elim_col]
        row = [det2(A[i][c], e1, A[i+1][c], e2) for c in cols]
        rhs = det2(b[i], e1, b[i+1], e2)
        A_new.append(row); b_new.append(rhs)
        log.append(f"  det2(row{i+1}, row{i+2}) → elim col {elim_col+1}  rhs={float(rhs):.4f}")
    return A_new, b_new, cols

def solve_kedm(A_in, b_in, var_names=None):
    n = len(b_in)
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]
    t0 = time.perf_counter()
    solutions = {}
    all_log = []
    total_det2 = 0

    all_log.append(f"KEDM Solver  —  {n}-variable system")
    all_log.append(f"Variables: {', '.join(var_names)}")
    all_log.append("─" * 50)

    for vi in range(n):
        A = [[Fraction(x) for x in row] for row in A_in]
        b = [Fraction(x) for x in b_in]
        cur_cols = list(range(n))
        log = [f"Solving for {var_names[vi]}:"]
        d2 = 0

        for ev in range(n):
            if ev == vi or ev not in cur_cols:
                continue
            ec = cur_cols.index(ev)
            A, b = _kedm_order_rows(A, b, ec)
            step_log = []
            A, b, rem = _kedm_eliminate(A, b, ec, step_log)
            d2 += len(step_log)
            log.append(f"  Eliminate {var_names[ev] if ev < len(var_names) else f'x{ev+1}'}  ({len(b)+1}→{len(b)} eqs)")
            log.extend(step_log)
            cur_cols = [cur_cols[i] for i in rem]

        if A and A[0][0] != 0:
            val = float(b[0] / A[0][0])
            solutions[var_names[vi]] = round(val, 8)
            log.append(f"  → {var_names[vi]} = {val:.6f}")
        else:
            solutions[var_names[vi]] = None
            log.append(f"  → Could not solve (singular?)")

        all_log.extend(log)
        all_log.append("")
        total_det2 += d2

    elapsed = (time.perf_counter() - t0) * 1000
    all_log.append(f"Total det2 calls: {total_det2}")
    all_log.append(f"Time: {elapsed:.4f} ms")

    return {
        "method": "KEDM",
        "solutions": solutions,
        "log": all_log,
        "elapsed_ms": round(elapsed, 6),
        "det2_calls": total_det2,
        "n": n,
    }


# ════════════════════════════════════════════
#  GAUSSIAN ELIMINATION
# ════════════════════════════════════════════

def solve_gaussian(A_in, b_in, var_names=None):
    n = len(b_in)
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]

    t0 = time.perf_counter()
    A = [row[:] + [b_in[i]] for i, row in enumerate(A_in)]
    log = [f"Gaussian Elimination  —  {n}-variable system", "─" * 50]

    # Forward elimination
    for col in range(n):
        # Partial pivoting
        max_row = max(range(col, n), key=lambda r: abs(A[r][col]))
        if max_row != col:
            A[col], A[max_row] = A[max_row], A[col]
            log.append(f"Swap row {col+1} ↔ row {max_row+1} (partial pivot)")

        if abs(A[col][col]) < 1e-12:
            log.append("ERROR: Zero pivot — system may be singular")
            break

        for row in range(col + 1, n):
            factor = A[row][col] / A[col][col]
            A[row] = [A[row][j] - factor * A[col][j] for j in range(n + 1)]
            log.append(f"  R{row+1} ← R{row+1} - ({factor:.4f}) × R{col+1}")

        log.append(f"  Pivot col {col+1} done")

    log.append("Upper triangular form achieved")
    log.append("Back substitution:")

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-12:
            x[i] = 0
            continue
        x[i] = A[i][n]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
        log.append(f"  {var_names[i]} = {x[i]:.6f}")

    elapsed = (time.perf_counter() - t0) * 1000
    solutions = {var_names[i]: round(x[i], 8) for i in range(n)}
    log.append(f"Time: {elapsed:.4f} ms")

    return {
        "method": "Gaussian Elimination",
        "solutions": solutions,
        "log": log,
        "elapsed_ms": round(elapsed, 6),
        "n": n,
    }


# ════════════════════════════════════════════
#  JACOBI ITERATION
# ════════════════════════════════════════════

def solve_jacobi(A_in, b_in, var_names=None, max_iter=100, tol=1e-8, x0=None):
    n = len(b_in)
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]
    t0 = time.perf_counter()

    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    log = [f"Jacobi Iteration  —  {n}-variable system",
           f"Initial guess: {x.tolist()}", "─" * 50]

    # Check diagonal dominance
    dom = all(abs(A[i, i]) >= sum(abs(A[i, j]) for j in range(n) if j != i) for i in range(n))
    log.append(f"Diagonally dominant: {'YES — guaranteed convergence' if dom else 'NO — may not converge'}")

    converged = False
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            if abs(A[i, i]) < 1e-14:
                log.append(f"ERROR: Zero diagonal at row {i+1} — cannot use Jacobi")
                break
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        err = np.max(np.abs(x_new - x))
        if k < 5 or k % 10 == 0:
            log.append(f"  Iter {k+1:3d}: {['{}={:.4f}'.format(var_names[i], x_new[i]) for i in range(n)]}  err={err:.2e}")

        x = x_new
        if err < tol:
            converged = True
            log.append(f"Converged at iteration {k+1}  (error={err:.2e})")
            break

    if not converged:
        log.append(f"WARNING: Did not converge in {max_iter} iterations. Results are approximate.")

    elapsed = (time.perf_counter() - t0) * 1000
    solutions = {var_names[i]: round(float(x[i]), 8) for i in range(n)}
    log.append(f"Time: {elapsed:.4f} ms")

    return {
        "method": "Jacobi Iteration",
        "solutions": solutions,
        "log": log,
        "elapsed_ms": round(elapsed, 6),
        "converged": converged,
        "n": n,
        "note": "Approximate solution" if not converged else "Converged",
    }


# ════════════════════════════════════════════
#  GAUSS-SEIDEL ITERATION
# ════════════════════════════════════════════

def solve_gauss_seidel(A_in, b_in, var_names=None, max_iter=100, tol=1e-8, x0=None):
    n = len(b_in)
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]
    t0 = time.perf_counter()

    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    log = [f"Gauss-Seidel Iteration  —  {n}-variable system",
           f"Initial guess: {x.tolist()}", "─" * 50]

    dom = all(abs(A[i, i]) >= sum(abs(A[i, j]) for j in range(n) if j != i) for i in range(n))
    log.append(f"Diagonally dominant: {'YES' if dom else 'NO — may not converge'}")

    converged = False
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            if abs(A[i, i]) < 1e-14:
                log.append(f"ERROR: Zero diagonal at row {i+1}")
                break
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i, i]

        err = np.max(np.abs(x - x_old))
        if k < 5 or k % 10 == 0:
            log.append(f"  Iter {k+1:3d}: {['{}={:.4f}'.format(var_names[i], x[i]) for i in range(n)]}  err={err:.2e}")

        if err < tol:
            converged = True
            log.append(f"Converged at iteration {k+1}  (error={err:.2e})")
            break

    if not converged:
        log.append(f"WARNING: Did not converge in {max_iter} iterations.")

    elapsed = (time.perf_counter() - t0) * 1000
    solutions = {var_names[i]: round(float(x[i]), 8) for i in range(n)}
    log.append(f"Time: {elapsed:.4f} ms")

    return {
        "method": "Gauss-Seidel",
        "solutions": solutions,
        "log": log,
        "elapsed_ms": round(elapsed, 6),
        "converged": converged,
        "n": n,
    }


# ════════════════════════════════════════════
#  CRAMER'S RULE
# ════════════════════════════════════════════

def solve_cramers(A_in, b_in, var_names=None):
    n = len(b_in)
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]
    t0 = time.perf_counter()

    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float)
    log = [f"Cramer's Rule  —  {n}-variable system", "─" * 50]

    det_A = np.linalg.det(A)
    log.append(f"det(A) = {det_A:.6f}")

    if abs(det_A) < 1e-12:
        log.append("ERROR: det(A) = 0 — system has no unique solution")
        return {"method": "Cramer's Rule", "solutions": {}, "log": log, "elapsed_ms": 0, "n": n}

    solutions = {}
    for i, var in enumerate(var_names):
        Ai = A.copy()
        Ai[:, i] = b
        det_Ai = np.linalg.det(Ai)
        val = det_Ai / det_A
        solutions[var] = round(val, 8)
        log.append(f"  det(A_{var}) = {det_Ai:.6f}  →  {var} = {det_Ai:.6f}/{det_A:.6f} = {val:.6f}")
        log.append(f"  (Required computing a {n}×{n} determinant for each variable)")

    log.append(f"NOTE: Cramer's Rule requires {n+1} full {n}×{n} determinant calculations")
    log.append(f"      KEDM only uses 2×2 determinants for the same system.")

    elapsed = (time.perf_counter() - t0) * 1000
    log.append(f"Time: {elapsed:.4f} ms")

    return {
        "method": "Cramer's Rule",
        "solutions": solutions,
        "log": log,
        "elapsed_ms": round(elapsed, 6),
        "det_A": float(det_A),
        "n": n,
    }


# ════════════════════════════════════════════
#  LU DECOMPOSITION
# ════════════════════════════════════════════

def solve_lu(A_in, b_in, var_names=None):
    n = len(b_in)
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]
    t0 = time.perf_counter()

    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float)
    log = [f"LU Decomposition  —  {n}-variable system", "─" * 50]

    try:
        # Pure numpy LU with partial pivoting
        U = A.copy(); L = np.eye(n); P = np.eye(n)
        for k in range(n):
            pivot = np.argmax(np.abs(U[k:, k])) + k
            if pivot != k:
                U[[k, pivot]] = U[[pivot, k]]
                P[[k, pivot]] = P[[pivot, k]]
                if k > 0: L[[k, pivot], :k] = L[[pivot, k], :k]
            for i in range(k+1, n):
                L[i,k] = U[i,k]/U[k,k]; U[i] -= L[i,k]*U[k]
        pb = P @ b
        y = np.zeros(n)
        for i in range(n): y[i] = pb[i] - np.dot(L[i,:i], y[:i])
        x = np.zeros(n)
        for i in range(n-1,-1,-1): x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:]))/U[i,i]
        log.append("LU factorisation: A = P·L·U (with partial pivoting)")
        log.append("Forward substitution (Ly = Pb) done")
        log.append("Back substitution  (Ux = y) done")
        for i, v in enumerate(var_names):
            log.append(f"  {v} = {x[i]:.6f}")
        solutions = {var_names[i]: round(float(x[i]), 8) for i in range(n)}
    except Exception as e:
        log.append(f"ERROR: {e}")
        solutions = {}

    elapsed = (time.perf_counter() - t0) * 1000
    log.append(f"Time: {elapsed:.4f} ms")

    return {
        "method": "LU Decomposition",
        "solutions": solutions,
        "log": log,
        "elapsed_ms": round(elapsed, 6),
        "n": n,
    }


# ════════════════════════════════════════════
#  QUADRATIC FORMULA
# ════════════════════════════════════════════

def solve_quadratic(text: str):
    text_c = text.strip().replace('**2', '^2').replace(' ', '')
    # Normalise: bring everything to LHS so = 0
    if '=' in text_c:
        lhs, rhs = text_c.split('=', 1)
        try:
            rhs_val = float(eval(rhs.replace('^', '**'), {"__builtins__": {}}))
        except:
            rhs_val = 0.0
        offset = -rhs_val
        text_c = lhs + ('+' if offset >= 0 else '') + str(offset)
    # text_c is now the LHS polynomial string (= 0 implicit)

    def parse_coef(s, default=1.0):
        s = s.strip()
        if not s or s == '+': return float(default)
        if s == '-': return -float(default)
        return float(s)

    # Extract a (coeff of x^2)
    a_m = re.search(r'([+-]?\d*\.?\d*)x\^2', text_c)
    if not a_m:
        raise ValueError("No x^2 term found. Use format: ax^2 + bx + c = 0")
    a = parse_coef(a_m.group(1))

    # Extract b (coeff of x, not x^2)
    b_m = re.search(r'([+-]?\d*\.?\d*)x(?!\^)', text_c)
    b = parse_coef(b_m.group(1)) if b_m else 0.0

    # Extract c: remove x^2 and x terms then evaluate remainder
    c_str = re.sub(r'[+-]?\d*\.?\d*x\^2', '', text_c)
    c_str = re.sub(r'[+-]?\d*\.?\d*x', '', c_str).strip()
    try:
        c = float(eval(c_str if c_str else '0', {"__builtins__": {}}))
    except:
        c = 0.0

    disc = b**2 - 4*a*c
    steps = [
        f"Standard form: {a}x² + {b}x + {c} = 0",
        f"Discriminant Δ = b² - 4ac = ({b})² - 4({a})({c}) = {disc}",
    ]
    if disc > 0:
        r1 = (-b + math.sqrt(disc)) / (2*a)
        r2 = (-b - math.sqrt(disc)) / (2*a)
        steps.append(f"Δ > 0 → two real roots")
        steps.append(f"x₁ = (-({b}) + √{disc}) / (2×{a}) = {r1:.6f}")
        steps.append(f"x₂ = (-({b}) - √{disc}) / (2×{a}) = {r2:.6f}")
        roots = [r1, r2]
    elif disc == 0:
        r = -b / (2*a)
        steps.append(f"Δ = 0 → one repeated root")
        steps.append(f"x = -{b} / (2×{a}) = {r:.6f}")
        roots = [r]
    else:
        re_part = -b / (2*a)
        im_part = math.sqrt(-disc) / (2*a)
        steps.append(f"Δ < 0 → two complex roots")
        steps.append(f"x = {re_part:.6f} ± {im_part:.6f}i")
        roots = [complex(re_part, im_part), complex(re_part, -im_part)]

    return {"roots": roots, "steps": steps, "a": a, "b": b, "c": c, "discriminant": disc}


# ════════════════════════════════════════════
#  ARITHMETIC EVALUATOR
# ════════════════════════════════════════════

def solve_arithmetic(expr: str):
    safe = {
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'round': round, 'pow': pow,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'log': math.log, 'log10': math.log10, 'exp': math.exp,
        'floor': math.floor, 'ceil': math.ceil,
    }
    clean = expr.strip().replace('^', '**')
    if clean.endswith('='): clean = clean[:-1]
    result = eval(clean, {"__builtins__": {}}, safe)
    return float(result)


# ════════════════════════════════════════════
#  PROBLEM TYPE CLASSIFIER
# ════════════════════════════════════════════

def classify_problem(text: str) -> str:
    t = text.lower().strip()
    lines = [l for l in t.split('\n') if l.strip()]

    if re.match(r'^[\d\s\+\-\*\/\(\)\.%\^]+$', t):
        return 'arithmetic'
    if ('^2' in t or '**2' in t) and '=' in t and len(lines) == 1:
        return 'quadratic'
    if len(lines) >= 2 and t.count('=') >= 2:
        return 'linear_system'
    if len(lines) == 1 and '=' in t:
        return 'single_linear'
    return 'linear_system'
