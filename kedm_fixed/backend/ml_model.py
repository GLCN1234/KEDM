"""
ml_model.py
===========
ML-powered equation parser and problem type classifier.
Uses scikit-learn to classify equation types, then routes to KEDM solver.
"""

import re
import numpy as np
from fractions import Fraction


# ─────────────────────────────────────────────
# EQUATION PARSER
# ─────────────────────────────────────────────

def parse_linear_system(text: str):
    """
    Parse a system of linear equations from natural text.
    Handles formats like:
        2x + 3y = 7
        -4x + y - 2z = 1
        5a - 3b + c + 2d = 10
    Returns (A, b, var_names) or raises ValueError.
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    equations = []

    # Collect all variable names in order of appearance
    all_vars = []
    var_pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9]*')

    for line in lines:
        # Remove spaces around = and split
        if '=' not in line:
            raise ValueError(f"No '=' found in equation: {line}")
        lhs, rhs = line.split('=', 1)
        vars_in_eq = var_pattern.findall(lhs)
        for v in vars_in_eq:
            if v not in all_vars:
                all_vars.append(v)

    n_vars = len(all_vars)
    if len(lines) != n_vars:
        raise ValueError(
            f"Expected {n_vars} equations for {n_vars} variables, got {len(lines)}"
        )

    A = []
    b = []
    for line in lines:
        lhs, rhs = line.split('=', 1)
        rhs_val = float(eval(rhs.strip()))
        coeffs = {v: 0.0 for v in all_vars}

        # Normalize: add spaces around + and - (not when part of number/start)
        lhs = re.sub(r'(?<=[^eE\d])-', ' -', lhs)
        lhs = re.sub(r'\+', ' + ', lhs)
        lhs = lhs.strip()

        # Match terms like: 3x, -2y, x, -z, 1.5abc
        term_pattern = re.compile(
            r'([+-]?\s*\d*\.?\d*)\s*([a-zA-Z][a-zA-Z0-9]*)'
        )
        for coeff_str, var in term_pattern.findall(lhs):
            coeff_str = coeff_str.replace(' ', '')
            if coeff_str in ('', '+'):
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)
            if var in coeffs:
                coeffs[var] += coeff

        A.append([coeffs[v] for v in all_vars])
        b.append(rhs_val)

    return A, b, all_vars


# ─────────────────────────────────────────────
# PROBLEM TYPE CLASSIFIER (ML)
# ─────────────────────────────────────────────

class EquationClassifier:
    """
    Lightweight ML classifier that detects what kind of math problem
    the user has entered and routes it to the correct solver.

    Uses rule-based feature extraction + a simple decision tree
    trained on synthetic examples.
    """

    PROBLEM_TYPES = [
        "linear_system",       # System of linear equations → KEDM
        "single_linear",       # One linear equation (1 unknown)
        "arithmetic",          # Pure arithmetic expression
        "cramers_demo",        # User explicitly mentions Cramer's rule
        "quadratic",           # ax^2 + bx + c = 0
    ]

    def __init__(self):
        self._build_model()

    def _extract_features(self, text: str):
        t = text.lower()
        lines = [l.strip() for l in t.split('\n') if l.strip()]
        n_lines = len(lines)
        n_equals = t.count('=')
        n_vars = len(set(re.findall(r'\b[a-z]\b', t)))
        has_cramer = int('cramer' in t)
        has_quad = int('^2' in t or '**2' in t or 'x2' in t)
        has_ops = int(any(op in t for op in ['+', '-', '*', '/']))
        only_numbers = int(bool(re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', t.strip())))
        multi_eq = int(n_lines > 1 and n_equals > 1)
        return np.array([
            n_lines, n_equals, n_vars, has_cramer,
            has_quad, has_ops, only_numbers, multi_eq
        ], dtype=float)

    def _build_model(self):
        """Train a simple decision tree on synthetic labeled examples."""
        try:
            from sklearn.tree import DecisionTreeClassifier
            # Synthetic training data: [n_lines, n_eq, n_vars, cramer, quad, ops, only_nums, multi_eq]
            X = np.array([
                # linear_system
                [2, 2, 2, 0, 0, 1, 0, 1],
                [3, 3, 3, 0, 0, 1, 0, 1],
                [4, 4, 4, 0, 0, 1, 0, 1],
                [2, 2, 2, 0, 0, 1, 0, 1],
                # single_linear
                [1, 1, 1, 0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 1, 0, 0],
                # arithmetic
                [1, 0, 0, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 1, 1, 0],
                [1, 1, 0, 0, 0, 1, 1, 0],
                # cramers_demo
                [2, 2, 2, 1, 0, 1, 0, 1],
                [3, 3, 3, 1, 0, 1, 0, 1],
                # quadratic
                [1, 1, 1, 0, 1, 1, 0, 0],
                [1, 1, 1, 0, 1, 1, 0, 0],
            ])
            y = [
                0, 0, 0, 0,  # linear_system
                1, 1,        # single_linear
                2, 2, 2,     # arithmetic
                3, 3,        # cramers_demo
                4, 4,        # quadratic
            ]
            self.clf = DecisionTreeClassifier(max_depth=5, random_state=42)
            self.clf.fit(X, y)
            self.ml_available = True
        except ImportError:
            self.ml_available = False

    def classify(self, text: str) -> str:
        """Return problem type string."""
        t = text.lower().strip()

        # Hard rules first (high confidence)
        if 'cramer' in t:
            return 'cramers_demo'
        if bool(re.match(r'^[\d\s\+\-\*\/\(\)\.%\^]+$', t)):
            return 'arithmetic'
        if ('^2' in t or '**2' in t) and '=' in t:
            return 'quadratic'

        lines = [l for l in t.split('\n') if l.strip()]
        if len(lines) >= 2 and t.count('=') >= 2:
            return 'linear_system'
        if len(lines) == 1 and '=' in t:
            return 'single_linear'

        if self.ml_available:
            feat = self._extract_features(text).reshape(1, -1)
            pred = self.clf.predict(feat)[0]
            return self.PROBLEM_TYPES[pred]

        return 'linear_system'


# ─────────────────────────────────────────────
# SPECIALIZED SOLVERS
# ─────────────────────────────────────────────

def solve_arithmetic(expr: str):
    """Evaluate pure arithmetic safely."""
    import operator, math
    safe_dict = {
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'round': round, 'pow': pow,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'log': math.log, 'log10': math.log10,
    }
    # Replace ^ with ** for exponentiation
    expr_clean = expr.strip().replace('^', '**')
    # Remove trailing = if any
    if expr_clean.endswith('='):
        expr_clean = expr_clean[:-1]
    result = eval(expr_clean, {"__builtins__": {}}, safe_dict)
    return float(result)


def solve_quadratic(text: str):
    """
    Solve ax^2 + bx + c = 0 using the quadratic formula.
    Returns dict with roots and discriminant.
    """
    import math
    # Extract coefficients
    text_clean = text.replace('**2', '^2').replace('x2', 'x^2')
    match = re.search(
        r'([+-]?\s*\d*\.?\d*)\s*x\^2\s*([+-]\s*\d*\.?\d*)\s*x\s*([+-]\s*\d*\.?\d*)\s*=\s*0',
        text_clean.replace(' ', '')
    )
    if not match:
        raise ValueError("Could not parse quadratic. Use format: ax^2 + bx + c = 0")

    a_s = match.group(1).replace(' ', '') or '1'
    b_s = match.group(2).replace(' ', '') or '0'
    c_s = match.group(3).replace(' ', '') or '0'
    a, b, c = float(a_s), float(b_s), float(c_s)

    disc = b**2 - 4*a*c
    steps = [
        f"Standard form: {a}x² + {b}x + {c} = 0",
        f"Discriminant: Δ = b² - 4ac = {b}² - 4({a})({c}) = {disc}",
    ]

    if disc > 0:
        r1 = (-b + math.sqrt(disc)) / (2 * a)
        r2 = (-b - math.sqrt(disc)) / (2 * a)
        steps.append(f"Two real roots: x₁ = {round(r1, 6)}, x₂ = {round(r2, 6)}")
        roots = [r1, r2]
    elif disc == 0:
        r = -b / (2 * a)
        steps.append(f"One repeated root: x = {round(r, 6)}")
        roots = [r]
    else:
        real = -b / (2 * a)
        imag = math.sqrt(-disc) / (2 * a)
        steps.append(f"Complex roots: x = {round(real, 6)} ± {round(imag, 6)}i")
        roots = [complex(real, imag), complex(real, -imag)]

    return {"roots": roots, "discriminant": disc, "steps": steps, "a": a, "b": b, "c": c}


def solve_single_linear(text: str):
    """
    Solve a single linear equation like: 3x + 5 = 14
    Converts to 1x1 KEDM system.
    """
    A, b, var_names = parse_linear_system(text)
    return A, b, var_names


def cramers_rule_solve(A, b, var_names=None):
    """
    Solve using Cramer's rule for comparison / demo purposes.
    Shows the determinant calculations step by step.
    """
    import time
    n = len(b)
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]

    t_start = time.perf_counter()
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)

    det_A = float(np.linalg.det(A_np))
    solutions = {}
    steps = [f"Main determinant |A| = {round(det_A, 6)}"]

    for i, var in enumerate(var_names):
        Ai = A_np.copy()
        Ai[:, i] = b_np
        det_Ai = float(np.linalg.det(Ai))
        val = det_Ai / det_A
        solutions[var] = val
        steps.append(f"|A_{var}| = {round(det_Ai, 6)}  →  {var} = {round(det_Ai,6)}/{round(det_A,6)} = {round(val, 6)}")

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    return {
        "solutions": solutions,
        "steps": steps,
        "det_A": det_A,
        "elapsed_ms": round(elapsed_ms, 4),
        "method": "Cramer's Rule",
    }
