import matplotlib.pyplot as plt
import numpy as np
import pulp
import streamlit as st

st.set_page_config(page_title="Programmation Linéaire - PuLP", layout="wide")
st.title("Résolution d'un PL à 2 variables (Ismael Badri)")

# --- 1. Paramètres dans la barre latérale ---
st.sidebar.header("Fonction objectif : Z = c1*x1 + c2*x2")
sens = st.sidebar.selectbox("Type d'optimisation", ["Maximiser", "Minimiser"])
c1 = st.sidebar.number_input("Coefficient c1 (pour x1)", value=3.0, step=0.5)
c2 = st.sidebar.number_input("Coefficient c2 (pour x2)", value=2.0, step=0.5)

st.sidebar.header("Contraintes : a1*x1 + a2*x2 [<= | >= | ==] b")
nb_contraintes = st.sidebar.slider("Nombre de contraintes", min_value=1, max_value=5, value=3)

contraintes = []
for i in range(nb_contraintes):
    st.sidebar.markdown(f"**Contrainte {i+1}**")
    col1, col2, col3, col4 = st.sidebar.columns([2, 2, 2, 2])
    a1 = col1.number_input(f"a1_{i+1}", value=1.0 if i == 0 else (2.0 if i == 1 else 1.0), step=0.5, key=f"a1_{i}")
    a2 = col2.number_input(f"a2_{i+1}", value=1.0 if i == 0 else (1.0 if i == 1 else 0.0), step=0.5, key=f"a2_{i}")
    signe = col3.selectbox(f"s_{i+1}", ["<=", ">=", "=="], key=f"s_{i}")
    b = col4.number_input(f"b_{i+1}", value=4.0 if i == 0 else (6.0 if i == 1 else 3.0), step=0.5, key=f"b_{i}")
    contraintes.append((a1, a2, signe, b))

# --- 2. Résolution avec PuLP (compatible PuLP 4.x et 3.x) ---
prob_sense = pulp.LpMaximize if sens == "Maximiser" else pulp.LpMinimize
prob = pulp.LpProblem("Probleme_Lineaire", prob_sense)

if hasattr(prob, "add_variable"):
    x1 = prob.add_variable("x1", lowBound=0, cat="Continuous")
    x2 = prob.add_variable("x2", lowBound=0, cat="Continuous")
else:
    x1 = pulp.LpVariable("x1", lowBound=0, cat="Continuous")
    x2 = pulp.LpVariable("x2", lowBound=0, cat="Continuous")

prob += c1 * x1 + c2 * x2, "Objectif"

for idx, (a1, a2, signe, b) in enumerate(contraintes):
    expr = a1 * x1 + a2 * x2
    if signe == "<=":
        prob += expr <= b, f"C_{idx+1}"
    elif signe == ">=":
        prob += expr >= b, f"C_{idx+1}"
    else:
        prob += expr == b, f"C_{idx+1}"

status_str = "Inconnu"
try:
    if hasattr(prob, "solve"):
        try:
            res = prob.solve(pulp.PULP_CBC_CMD(msg=False))
            status_str = pulp.LpStatus[res] if hasattr(pulp, "LpStatus") else str(res)
        except Exception:
            res = prob.solve()
            status_str = getattr(res, "status_str", str(res))
except Exception as e:
    status_str = f"Erreur solveur : {e}"

# Fallback léger avec scipy.optimize.linprog si PuLP n'a aucun solveur actif
used_scipy = False
val_x1, val_x2, val_z = None, None, None

if status_str.lower() == "optimal":
    val_x1 = pulp.value(x1)
    val_x2 = pulp.value(x2)
    val_z = pulp.value(prob.objective)
elif "No solver available" in status_str:
    try:
        from scipy.optimize import linprog

        obj = [-c1, -c2] if sens == "Maximiser" else [c1, c2]
        A_ub, b_ub = [], []
        A_eq, b_eq = [], []

        for a1, a2, signe, b in contraintes:
            if signe == "<=":
                A_ub.append([a1, a2])
                b_ub.append(b)
            elif signe == ">=":
                A_ub.append([-a1, -a2])
                b_ub.append(-b)
            else:
                A_eq.append([a1, a2])
                b_eq.append(b)

        lp_res = linprog(
            c=obj,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq if A_eq else None,
            b_eq=b_eq if b_eq else None,
            bounds=[(0, None), (0, None)],
            method="highs",
        )
        if lp_res.success:
            status_str = "Optimal"
            used_scipy = True
            val_x1 = float(lp_res.x[0])
            val_x2 = float(lp_res.x[1])
            val_z = float(c1 * val_x1 + c2 * val_x2)
    except Exception:
        pass

# --- 3. Affichage des résultats ---
st.subheader("Résultat de l'optimisation")
if status_str.lower() == "optimal" and val_x1 is not None and val_x2 is not None:
    msg_extra = " (via HiGHS / SciPy)" if used_scipy else ""
    st.success(f"Statut : **Optimal**{msg_extra}")
    st.write(f"- **$x_1^*$** = `{val_x1:.4f}`")
    st.write(f"- **$x_2^*$** = `{val_x2:.4f}`")
    st.write(f"- **Valeur optimale de $Z$** = `{val_z:.4f}`")
else:
    st.warning(f"Statut : **{status_str}**")
    val_x1, val_x2, val_z = None, None, None

# --- 4. Approche graphique ---
st.subheader("Approche graphique")

bornes_b = [abs(b) for _, _, _, b in contraintes if b != 0]
borne_max = max(bornes_b) * 1.3 if bornes_b else 10.0
if val_x1 is not None and val_x2 is not None:
    borne_max = max(borne_max, val_x1 * 1.3, val_x2 * 1.3, 5.0)

grille_1d = np.linspace(0, borne_max, 400)
X1, X2 = np.meshgrid(grille_1d, grille_1d)

faisable = np.ones_like(X1, dtype=bool)
tol = 1e-6

fig, ax = plt.subplots(figsize=(7, 5))

for idx, (a1, a2, signe, b) in enumerate(contraintes):
    lhs = a1 * X1 + a2 * X2
    if signe == "<=":
        faisable &= (lhs <= b + tol)
    elif signe == ">=":
        faisable &= (lhs >= b - tol)
    else:
        faisable &= (np.abs(lhs - b) <= 0.05)

    label_c = f"C{idx+1}: {a1}x1 + {a2}x2 {signe} {b}"
    if abs(a2) > 1e-9:
        y_line = (b - a1 * grille_1d) / a2
        ax.plot(grille_1d, y_line, label=label_c)
    elif abs(a1) > 1e-9:
        x_line = np.full_like(grille_1d, b / a1)
        ax.plot(x_line, grille_1d, label=label_c)

ax.imshow(
    faisable.astype(int),
    extent=(0, borne_max, 0, borne_max),
    origin="lower",
    cmap="Blues",
    alpha=0.25,
    aspect="auto",
)

if status_str.lower() == "optimal" and val_x1 is not None and val_x2 is not None:
    ax.scatter([val_x1], [val_x2], color="red", zorder=5, s=70, label=f"Optimum ({val_x1:.2f}, {val_x2:.2f})")
    if abs(c2) > 1e-9:
        z_line = (val_z - c1 * grille_1d) / c2
        ax.plot(grille_1d, z_line, "--", color="red", label=f"Objectif Z = {val_z:.2f}")
    elif abs(c1) > 1e-9:
        ax.axvline(val_z / c1, linestyle="--", color="red", label=f"Objectif Z = {val_z:.2f}")

ax.set_xlim(0, borne_max)
ax.set_ylim(0, borne_max)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Région réalisable et solution optimale")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="best", fontsize=8)

st.pyplot(fig)
