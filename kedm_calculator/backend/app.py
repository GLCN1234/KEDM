"""
app.py  —  KEDM Calculator API
"""
import json, time
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from solvers import (
    parse_linear_system, classify_problem,
    solve_kedm, solve_gaussian, solve_jacobi,
    solve_gauss_seidel, solve_cramers, solve_lu,
    solve_quadratic, solve_arithmetic,
)

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

SOLVER_MAP = {
    "kedm":         solve_kedm,
    "gaussian":     solve_gaussian,
    "jacobi":       solve_jacobi,
    "gauss_seidel": solve_gauss_seidel,
    "cramer":       solve_cramers,
    "lu":           solve_lu,
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve():
    data = request.get_json() or {}
    user_input = data.get('input', '').strip()
    method     = data.get('method', 'kedm')
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    try:
        problem_type = classify_problem(user_input)
        result = _dispatch(user_input, problem_type, method)
        result['problem_type'] = problem_type
        result['input'] = user_input
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/solve/stream', methods=['POST'])
def solve_stream():
    data = request.get_json() or {}
    user_input = data.get('input', '').strip()
    method     = data.get('method', 'kedm')

    def generate():
        try:
            problem_type = classify_problem(user_input)
            yield _sse({"type": "classify", "problem_type": problem_type, "method": method})
            result = _dispatch(user_input, problem_type, method)
            for line in result.get('log', []):
                yield _sse({"type": "log", "line": line})
                time.sleep(0.035)
            yield _sse({
                "type": "done",
                "solutions": result.get('solutions', {}),
                "elapsed_ms": result.get('elapsed_ms', 0),
                "det2_calls": result.get('det2_calls', 0),
                "method": result.get('method', method),
                "note": result.get('note', ''),
                "problem_type": problem_type,
            })
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/compare', methods=['POST'])
def compare_all():
    data = request.get_json() or {}
    user_input = data.get('input', '').strip()
    try:
        A, b, var_names = parse_linear_system(user_input)
        results = {}
        for name, fn in SOLVER_MAP.items():
            try:
                r = fn(A, b, var_names)
                results[name] = {
                    "solutions": r['solutions'],
                    "elapsed_ms": r['elapsed_ms'],
                    "method": r['method'],
                    "det2_calls": r.get('det2_calls', '—'),
                    "note": r.get('note', ''),
                }
            except Exception as e:
                results[name] = {"error": str(e)}
        return jsonify({"comparison": results, "variables": var_names, "n": len(var_names)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/examples')
def examples():
    return jsonify([
        {"label": "2-Variable (Paper)", "input": "4R1 - 7R2 = -50\n-6R1 + 9R2 = 60", "type": "linear_system", "expected": "R1=5, R2=10"},
        {"label": "3-Variable (Paper)", "input": "7i1 + 3i2 + 4i3 = 29\n-8i1 + 5i2 - 6i3 = -19\n2i1 - i2 - 9i3 = 47", "type": "linear_system", "expected": "i1=6, i2=1, i3=-4"},
        {"label": "4-Variable Zero Coefficients", "input": "-4m1 - 3m2 + 7m3 + 2m4 = 5\n-5m1 + 8m3 + 6m4 = 119\n-13m1 + 9m2 + 10m4 = 240\n-m1 + 12m2 - 5m3 - 3m4 = 211", "type": "linear_system", "expected": "m1=5,m2=25,m3=12,m4=8"},
        {"label": "Cramer's Rule Demo", "input": "2x + 3y = 13\n5x - y = 2", "type": "linear_system", "expected": "x=1, y≈3.67"},
        {"label": "Gaussian Elimination", "input": "2a + b - c = 8\n-3a - b + 2c = -11\n-2a + b + 2c = -3", "type": "linear_system", "expected": "a=2,b=3,c=-1"},
        {"label": "Jacobi / Gauss-Seidel", "input": "10x + y + z = 12\nx + 10y - z = 10\n-x + y + 10z = 10", "type": "linear_system", "expected": "x=y=z=1"},
        {"label": "5-Variable Extended", "input": "2a + b - c + 3d - e = 10\na - 2b + 4c - d + 2e = -5\n3a + 2b - c + d - 3e = 8\n-a + b + 2c + 4d + e = 12\n4a - b + c - 2d + 3e = 6", "type": "linear_system", "expected": "KEDM extension"},
        {"label": "Kirchhoff's Circuit", "input": "2I1 - I2 + 3I3 = 12\nI1 + 4I2 - I3 = 5\n3I1 - 2I2 + I3 = 8", "type": "linear_system", "expected": "Circuit currents"},
        {"label": "Quadratic Formula", "input": "2x^2 - 4x - 6 = 0", "type": "quadratic", "expected": "x=3, x=-1"},
        {"label": "Complex Arithmetic", "input": "((25 * 4) + sqrt(144)) / 2 - 37", "type": "arithmetic", "expected": "≈19"},
    ])

def _dispatch(text, problem_type, method):
    if problem_type == 'arithmetic':
        val = solve_arithmetic(text)
        return {"method": "arithmetic", "solutions": {"result": val},
                "log": [f"Expression: {text}", f"Result: {val}"], "elapsed_ms": 0}
    if problem_type == 'quadratic':
        q = solve_quadratic(text)
        return {"method": "quadratic_formula",
                "solutions": {f"root_{i+1}": str(r) for i, r in enumerate(q['roots'])},
                "log": q['steps'], "elapsed_ms": 0, "discriminant": q.get('discriminant')}
    A, b, var_names = parse_linear_system(text)
    solver_fn = SOLVER_MAP.get(method, solve_kedm)
    return solver_fn(A, b, var_names)

def _sse(data):
    return f"data: {json.dumps(data)}\n\n"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
