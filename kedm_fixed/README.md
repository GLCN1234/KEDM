# KEDM Calculator

Solves equations using KEDM, Gaussian, Jacobi, Gauss-Seidel, Cramer's Rule, LU Decomposition, Quadratic Formula, and Arithmetic — with a live terminal showing every step.

Based on: Osanyinpeju, K.L. (2025). JSID 7(1). DOI: 10.20372/jsid/2025-340

---

## How to enter your own equations

**Linear system (one per line):**
```
4x + 2y = 20
3x - y  = 5
```
Any variable names work: R1, I1, a, m1, etc.
Zero-coefficient terms can just be omitted.

**Quadratic:**  `2x^2 - 4x - 6 = 0`

**Arithmetic:**  `((25 * 4) + sqrt(144)) / 2 - 37`

**Keyboard shortcut:** Ctrl+Enter to solve

---

## GITHUB CODESPACES — unzip and run

```bash
unzip kedm_calculator.zip
cd kedm_calculator
pip install -r requirements.txt
python backend/app.py
```
Click "Open in Browser" when Codespaces prompts you. App runs on port 5000.

---

## RENDER DEPLOYMENT (free Python hosting)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "KEDM Calculator"
git remote add origin https://github.com/YOUR_USERNAME/kedm-calculator.git
git branch -M main
git push -u origin main
```

### Step 2 — Create Render account
Go to https://render.com — sign up with GitHub.

### Step 3 — New Web Service
1. Click New + → Web Service
2. Connect your kedm-calculator repo

### Step 4 — Fill in these settings

| Field            | Value                                                     |
|------------------|-----------------------------------------------------------|
| Name             | kedm-calculator                                           |
| Branch           | main                                                      |
| Root Directory   | kedm_calculator                                           |
| Runtime          | Python 3                                                  |
| Build Command    | pip install -r requirements.txt                           |
| Start Command    | gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT     |
| Instance Type    | Free                                                      |

### Step 5 — Click Create Web Service
Wait ~2 minutes. Your app will be live at:
  https://kedm-calculator.onrender.com

### Step 6 — Update and redeploy any time
```bash
git add .
git commit -m "your changes"
git push
# Render redeploys automatically
```

---

## Solvers available

- KEDM (foundation, 2x2 det only, works for any n)
- Gaussian Elimination (with partial pivoting)
- Jacobi Iteration
- Gauss-Seidel Iteration
- Cramer's Rule
- LU Decomposition
- Quadratic Formula
- Arithmetic evaluator (sqrt, sin, cos, log, etc.)

---

## Project structure

```
kedm_calculator/
├── backend/
│   ├── app.py         Flask API + streaming
│   ├── kedm_engine.py KEDM core math
│   ├── ml_model.py    ML classifier
│   └── solvers.py     All solvers
├── frontend/templates/index.html
├── requirements.txt
└── Procfile
```
