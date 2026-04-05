import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import pearsonr
import os
import tensorflow as tf
keras = tf.keras

st.set_page_config(
    page_title="Heap Leach ML Predictor",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams.update({
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.labelsize":        10,
    "xtick.labelsize":        9,
    "ytick.labelsize":        9,
    "legend.fontsize":        9,
    "legend.title_fontsize":  9,
    "figure.titlesize":      12,
})

FOLDER = os.path.dirname(os.path.abspath(__file__))

st.markdown("""
<style>
    /* Global body text */
    html, body, [class*="css"] {
        font-size: 22px !important;
    }

    /* Main content text */
    .stMarkdown, .stText, p, li, span, label {
        font-size: 22px !important;
    }

    /* Headings */
    h1 { font-size: 52px !important; font-weight: 700 !important; }
    h2 { font-size: 40px !important; font-weight: 600 !important; }
    h3 { font-size: 32px !important; font-weight: 600 !important; }

    /* st.title specifically */
    [data-testid="stAppViewContainer"] h1,
    .css-10trblm, .css-1629p8f {
        font-size: 52px !important;
        font-weight: 700 !important;
    }

    /* st.markdown ### subheadings */
    [data-testid="stAppViewContainer"] h3 {
        font-size: 32px !important;
        font-weight: 600 !important;
    }

    /* st.caption */
    .stCaption, [data-testid="stCaptionContainer"] p,
    small, .css-1fttcpj {
        font-size: 18px !important;
    }

    /* bold inline titles e.g. **Feature importance** */
    strong {
        font-size: 24px !important;
        font-weight: 600 !important;
    }

    /* Sidebar labels and text */
    .css-1544g2n, .css-ng1t4o,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        font-size: 20px !important;
    }

    /* Slider labels */
    .stSlider label, .stSlider p {
        font-size: 20px !important;
    }

    /* Metric cards — value and label */
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 20px !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 18px !important;
    }

    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        padding: 12px 24px !important;
    }

    /* Selectbox and input text */
    .stSelectbox label, .stTextInput label,
    .stTextArea label, .stCheckbox label {
        font-size: 20px !important;
    }
    .stSelectbox div[data-baseweb="select"] span,
    .stTextInput input, .stTextArea textarea {
        font-size: 20px !important;
    }

    /* Buttons */
    .stButton button {
        font-size: 20px !important;
        padding: 12px 28px !important;
    }

    /* Dataframe/table text */
    .stDataFrame, .dataframe td, .dataframe th {
        font-size: 18px !important;
    }

    /* Caption text */
    .stCaption, caption {
        font-size: 18px !important;
    }

    /* Info/warning/success boxes */
    .stAlert p {
        font-size: 20px !important;
    }

    /* Download button */
    .stDownloadButton button {
        font-size: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    sX = joblib.load(os.path.join(FOLDER, "scaler_X.pkl"))
    sy = joblib.load(os.path.join(FOLDER, "scaler_y.pkl"))
    m = keras.models.load_model(os.path.join(FOLDER, "ann_model.h5"), compile=False)
    return sX, sy, m

@st.cache_resource
def load_rf():
    try:
        return joblib.load(os.path.join(FOLDER, "random_forest_model.pkl"))
    except:
        return None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(os.path.join(FOLDER, "comsol_results_fixed.csv"))
    except:
        return None

scaler_X, scaler_y, model = load_models()
rf_model = load_rf()
df_data  = load_dataset()

TIME_DAYS   = [1, 3, 7, 14, 21, 30, 45, 60]
TIME_LABELS = ["1d","3d","7d","14d","21d","30d","45d","60d"]
FEATURES    = ["por","C0","u0","k_r","D_eff"]
FEAT_LABELS = ["Porosity φ","Acid C₀","Darcy u₀","Rate k_r","Diffusivity D_eff"]

PARAM_RANGES = {
    "por":   (0.25, 0.50),
    "C0":    (0.10, 1.00),
    "u0":    (1e-5, 1e-4),
    "k_r":   (1e-8, 1e-6),
    "D_eff": (1e-10, 1e-9),
}

def predict(por, C0, u0, kr, de):
    vec = np.array([[por, C0, u0, kr, de]])
    sc  = scaler_X.transform(vec)
    out = scaler_y.inverse_transform(model.predict(sc, verbose=0))
    return np.clip(out[0], 0, 1)

def badge(v):
    p = v * 100
    if p > 15: return "🟢 high"
    if p > 7:  return "🟡 moderate"
    return "🔴 low"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⛏️ Heap Leach Extraction Yield Predictor")
st.caption("ANN surrogate model · trained on 200 COMSOL simulations · R² = 0.9902 · Physics-ML hybrid workflow")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Scenario A parameters")
por  = st.sidebar.slider("Porosity (por)",                    0.25, 0.50, 0.35, 0.01)
C0   = st.sidebar.slider("Inlet acid C₀ (mol/m³)",           0.10, 1.00, 0.50, 0.01)
u0   = st.sidebar.slider("Darcy velocity u₀ (×10⁻⁵ m/s)",   1.0,  10.0, 5.0,  0.1)
kr   = st.sidebar.slider("Reaction rate k_r (×10⁻⁷ m/s)",   0.1,  10.0, 5.0,  0.1)
de   = st.sidebar.slider("Diffusivity D_eff (×10⁻¹⁰ m²/s)", 1.0,  10.0, 4.0,  0.1)

u0_v = u0 * 1e-5
kr_v = kr * 1e-7
de_v = de * 1e-10

pred_A = predict(por, C0, u0_v, kr_v, de_v)

st.sidebar.divider()
st.sidebar.markdown("**Model info**")
st.sidebar.caption("ANN: 5→128→128→64→32→8")
st.sidebar.caption("Trained on LHS-sampled COMSOL data")
st.sidebar.caption("R² = 0.9902 | RMSE = 0.0048")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Predict",
    "📊 Sensitivity",
    "🔀 Compare scenarios",
    "🎯 Inverse design",
    "🔬 Dataset explorer",
    "📋 Export report"
])

# ══════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("X at  7 days", f"{pred_A[2]*100:.2f}%", badge(pred_A[2]))
    c2.metric("X at 14 days", f"{pred_A[3]*100:.2f}%", badge(pred_A[3]))
    c3.metric("X at 30 days", f"{pred_A[5]*100:.2f}%", badge(pred_A[5]))
    c4.metric("X at 60 days", f"{pred_A[7]*100:.2f}%", badge(pred_A[7]))
    st.divider()

    col_plot, col_feat = st.columns([2, 1])

    with col_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(TIME_DAYS, pred_A*100, "o-", color="#378ADD",
                linewidth=2.5, markersize=7, label="ANN prediction")
        ax.fill_between(TIME_DAYS, pred_A*100*0.93, pred_A*100*1.07,
                        alpha=0.12, color="#378ADD", label="±7% band")
        ax.set_xlabel("Time (days)"); ax.set_ylabel("Extraction yield X (%)")
        ax.set_title("Predicted extraction yield over 60 days")
        ax.set_xlim(0, 63); ax.grid(True, alpha=0.25); ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_feat:
        st.markdown("**Feature importance (RF)**")
        imps = rf_model.feature_importances_ if rf_model else np.array([0.014, 0.386, 0.332, 0.252, 0.017])
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        colors_imp = ["#D85A30","#1D9E75","#378ADD","#7F77DD","#BA7517"]
        ax2.barh(FEAT_LABELS, imps, color=colors_imp, edgecolor="none")
        ax2.set_xlabel("Importance score")
        ax2.set_title("RF feature importance")
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.divider()

    # Leaching rate (dX/dt)
    st.markdown("**Instantaneous leaching rate (dX/dt)**")
    rates = np.diff(pred_A*100) / np.diff(TIME_DAYS)
    mid_times = [(TIME_DAYS[i]+TIME_DAYS[i+1])/2 for i in range(len(TIME_DAYS)-1)]
    fig_rate, ax_rate = plt.subplots(figsize=(8, 3))
    ax_rate.bar(mid_times, rates, width=2.5, color="#7F77DD", alpha=0.8, edgecolor="none")
    ax_rate.set_xlabel("Time (days)"); ax_rate.set_ylabel("dX/dt (% per day)")
    ax_rate.set_title("Leaching rate over time — peak rate identifies optimal harvest window")
    ax_rate.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    st.pyplot(fig_rate); plt.close()

    peak_idx = np.argmax(rates)
    st.info(f"Peak leaching rate: **{rates[peak_idx]:.3f}% per day** between days {TIME_DAYS[peak_idx]}–{TIME_DAYS[peak_idx+1]}")

    st.divider()
    st.markdown("**All time points**")
    tbl = pd.DataFrame({
        "Time":         TIME_LABELS,
        "X yield":      [f"{v*100:.3f}%" for v in pred_A],
        "Cumulative Δ": [f"+{(v - pred_A[0])*100:.3f}%" if i>0 else "—" for i,v in enumerate(pred_A)],
        "Status":       [badge(v) for v in pred_A]
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 2 — SENSITIVITY
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("### One-at-a-time sensitivity at day 60")
    st.caption("Each parameter swept ×0.5 to ×1.5 of current value, others fixed")

    base = [por, C0, u0_v, kr_v, de_v]
    swings = []
    for i in range(5):
        lo = base.copy(); hi = base.copy()
        lo[i] *= 0.5;    hi[i] *= 1.5
        lo[i] = max(lo[i], list(PARAM_RANGES.values())[i][0])
        hi[i] = min(hi[i], list(PARAM_RANGES.values())[i][1])
        swings.append(predict(*hi)[7] - predict(*lo)[7])

    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    bar_colors = ["#1D9E75" if s >= 0 else "#A32D2D" for s in swings]
    ax3.barh(FEAT_LABELS, [s*100 for s in swings], color=bar_colors, edgecolor="none")
    ax3.axvline(0, color="gray", linewidth=0.8)
    ax3.set_xlabel("Change in X_60d (percentage points)")
    ax3.set_title("Sensitivity tornado chart — current parameter set")
    plt.tight_layout()
    st.pyplot(fig3); plt.close()

    st.divider()
    st.markdown("### Full sweep: one parameter vs time")
    sweep_param = st.selectbox("Select parameter to sweep", FEAT_LABELS, key="sweep_sel")
    param_key   = FEATURES[FEAT_LABELS.index(sweep_param)]

    n_sweep = 8
    lo_r, hi_r = PARAM_RANGES[param_key]
    sweep_vals  = np.linspace(lo_r, hi_r, n_sweep)

    fig_sw, ax_sw = plt.subplots(figsize=(9, 4))
    cmap = plt.cm.viridis(np.linspace(0, 1, n_sweep))
    for idx, val in enumerate(sweep_vals):
        params = [por, C0, u0_v, kr_v, de_v]
        params[FEATURES.index(param_key)] = val
        y_sw = predict(*params) * 100
        label_val = f"{val:.2e}" if val < 1e-4 else f"{val:.3f}"
        ax_sw.plot(TIME_DAYS, y_sw, "o-", color=cmap[idx],
                   linewidth=1.8, markersize=4, label=label_val)

    ax_sw.set_xlabel("Time (days)"); ax_sw.set_ylabel("Extraction yield (%)")
    ax_sw.set_title(f"Effect of {sweep_param} across full range")
    ax_sw.legend(title=param_key, bbox_to_anchor=(1.01, 1), loc="upper left",
                 fontsize=8, title_fontsize=9)
    ax_sw.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_sw); plt.close()

    st.divider()
    st.markdown("### C₀ × u₀ interaction heatmap → X at 60 days")
    C0_vals = np.linspace(0.1, 1.0, 10)
    u0_vals = np.linspace(1e-5, 1e-4, 10)
    heat = np.zeros((len(C0_vals), len(u0_vals)))
    for i, c in enumerate(C0_vals):
        for j, u in enumerate(u0_vals):
            heat[i, j] = predict(por, c, u, kr_v, de_v)[7] * 100

    fig4, ax4 = plt.subplots(figsize=(9, 4.5))
    im = ax4.imshow(heat, aspect="auto", origin="lower", cmap="YlGn",
                    extent=[u0_vals[0]*1e5, u0_vals[-1]*1e5, C0_vals[0], C0_vals[-1]])
    cs = ax4.contour(np.linspace(u0_vals[0]*1e5, u0_vals[-1]*1e5, 10),
                     np.linspace(C0_vals[0], C0_vals[-1], 10),
                     heat, levels=6, colors="white", linewidths=0.8, alpha=0.6)
    ax4.clabel(cs, fmt="%.1f%%", fontsize=8)
    ax4.set_xlabel("u₀ (×10⁻⁵ m/s)"); ax4.set_ylabel("C₀ (mol/m³)")
    ax4.set_title("X_60d (%) heatmap — C₀ vs u₀ interaction (with contours)")
    plt.colorbar(im, ax=ax4, label="X_60d (%)")
    plt.tight_layout()
    st.pyplot(fig4); plt.close()

    st.divider()
    st.markdown("### Time to reach target yield")
    target = st.slider("Target yield (%)", 2, 40, 15, key="target_t2") / 100
    d = next((TIME_DAYS[i] for i, v in enumerate(pred_A) if v >= target), None)
    if d:
        st.success(f"✅ Target {target*100:.0f}% reached by **day {d}**")
    else:
        st.warning(f"⚠️ Target {target*100:.0f}% not reached within 60 days — max = {pred_A[7]*100:.1f}%")


# ══════════════════════════════════════════════════════
# TAB 3 — COMPARE SCENARIOS
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("### Multi-scenario comparison (A vs B vs C)")
    st.caption("Scenario A = sidebar. Set B and C below.")

    col_b, col_c = st.columns(2)
    with col_b:
        st.markdown("**Scenario B**")
        b_C0  = st.slider("B: C₀ (mol/m³)",          0.10, 1.00, 0.80, 0.01, key="bC0")
        b_u0  = st.slider("B: u₀ (×10⁻⁵ m/s)",       1.0,  10.0, 8.0,  0.1,  key="bu0")
        b_por = st.slider("B: Porosity",               0.25, 0.50, 0.40, 0.01, key="bpor")
        b_kr  = st.slider("B: k_r (×10⁻⁷ m/s)",      0.1,  10.0, 5.0,  0.1,  key="bkr")
        b_de  = st.slider("B: D_eff (×10⁻¹⁰ m²/s)",  1.0,  10.0, 4.0,  0.1,  key="bde")

    with col_c:
        st.markdown("**Scenario C**")
        c_C0  = st.slider("C: C₀ (mol/m³)",          0.10, 1.00, 0.30, 0.01, key="cC0")
        c_u0  = st.slider("C: u₀ (×10⁻⁵ m/s)",       1.0,  10.0, 3.0,  0.1,  key="cu0")
        c_por = st.slider("C: Porosity",               0.25, 0.50, 0.30, 0.01, key="cpor")
        c_kr  = st.slider("C: k_r (×10⁻⁷ m/s)",      0.1,  10.0, 2.0,  0.1,  key="ckr")
        c_de  = st.slider("C: D_eff (×10⁻¹⁰ m²/s)",  1.0,  10.0, 4.0,  0.1,  key="cde")

    pred_B = predict(b_por, b_C0, b_u0*1e-5, b_kr*1e-7, b_de*1e-10)
    pred_C = predict(c_por, c_C0, c_u0*1e-5, c_kr*1e-7, c_de*1e-10)

    fig5, axes5 = plt.subplots(1, 2, figsize=(13, 4.5))

    # Yield curves
    axes5[0].plot(TIME_DAYS, pred_A*100, "o-",  color="#378ADD", lw=2.5, ms=6,
                  label=f"A: C₀={C0:.2f}  u₀={u0:.1f}")
    axes5[0].plot(TIME_DAYS, pred_B*100, "s--", color="#1D9E75", lw=2.5, ms=6,
                  label=f"B: C₀={b_C0:.2f}  u₀={b_u0:.1f}")
    axes5[0].plot(TIME_DAYS, pred_C*100, "^:",  color="#D85A30", lw=2.5, ms=6,
                  label=f"C: C₀={c_C0:.2f}  u₀={c_u0:.1f}")
    axes5[0].fill_between(TIME_DAYS,
                          np.minimum(np.minimum(pred_A, pred_B), pred_C)*100,
                          np.maximum(np.maximum(pred_A, pred_B), pred_C)*100,
                          alpha=0.06, color="#7F77DD", label="Scenario range")
    axes5[0].set_xlabel("Time (days)"); axes5[0].set_ylabel("X yield (%)")
    axes5[0].set_title("Scenario comparison — yield curves")
    axes5[0].legend(fontsize=9); axes5[0].grid(True, alpha=0.25)

    # Final yield bar
    final_yields = [pred_A[7]*100, pred_B[7]*100, pred_C[7]*100]
    bar_cols5 = ["#378ADD", "#1D9E75", "#D85A30"]
    bars5 = axes5[1].bar(["Scenario A","Scenario B","Scenario C"],
                          final_yields, color=bar_cols5, edgecolor="none", width=0.5)
    for bar, val in zip(bars5, final_yields):
        axes5[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                      f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axes5[1].set_ylabel("X_60d (%)"); axes5[1].set_title("Final yield comparison at 60 days")
    axes5[1].grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    st.pyplot(fig5); plt.close()

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    best = max([(pred_A[7],"A"),(pred_B[7],"B"),(pred_C[7],"C")], key=lambda x: x[0])
    m1.metric("A — X 60d", f"{pred_A[7]*100:.2f}%")
    m2.metric("B — X 60d", f"{pred_B[7]*100:.2f}%")
    m3.metric("C — X 60d", f"{pred_C[7]*100:.2f}%")
    m4.metric("Best scenario", f"Scenario {best[1]}", f"{best[0]*100:.2f}%")

    cmp_df = pd.DataFrame({
        "Time":  TIME_LABELS,
        "A (%)": [f"{v*100:.3f}" for v in pred_A],
        "B (%)": [f"{v*100:.3f}" for v in pred_B],
        "C (%)": [f"{v*100:.3f}" for v in pred_C],
        "B−A":   [f"{(b-a)*100:+.3f}" for a,b in zip(pred_A, pred_B)],
        "C−A":   [f"{(c-a)*100:+.3f}" for a,c in zip(pred_A, pred_C)],
    })
    st.dataframe(cmp_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 4 — INVERSE DESIGN
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown("### Inverse design — find parameters that achieve a target yield")
    st.caption("Uses scipy.optimize to search the parameter space. No COMSOL needed.")

    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        target_yield = st.slider("Target extraction yield (%)", 5, 35, 20) / 100
        target_day   = st.selectbox("At which time point?",
                                    TIME_LABELS, index=7, key="inv_day")
        t_idx = TIME_LABELS.index(target_day)

    with col_inv2:
        st.markdown("**Fix these parameters (leave others free to optimise)**")
        fix_por = st.checkbox("Fix porosity", value=True)
        fix_de  = st.checkbox("Fix D_eff",    value=True)

    if st.button("🔍 Find optimal parameters", type="primary"):
        with st.spinner("Searching parameter space..."):

            def objective(x):
                # x = [C0, u0, kr] (or all 5 depending on fixing)
                if fix_por and fix_de:
                    params = [por, x[0], x[1]*1e-5, x[2]*1e-7, de_v]
                else:
                    params = [x[0], x[1], x[2]*1e-5, x[3]*1e-7, x[4]*1e-10]
                y = predict(*params)
                return (y[t_idx] - target_yield)**2

            results_list = []
            # Multi-start to avoid local minima
            for _ in range(12):
                if fix_por and fix_de:
                    x0     = [np.random.uniform(0.1, 1.0),
                               np.random.uniform(1, 10),
                               np.random.uniform(0.1, 10)]
                    bounds = [(0.1, 1.0), (1.0, 10.0), (0.1, 10.0)]
                else:
                    x0     = [np.random.uniform(0.25, 0.50),
                               np.random.uniform(0.1,  1.0),
                               np.random.uniform(1,    10),
                               np.random.uniform(0.1,  10),
                               np.random.uniform(1,    10)]
                    bounds = [(0.25,0.50),(0.1,1.0),(1,10),(0.1,10),(1,10)]

                res = minimize(objective, x0, method="L-BFGS-B",
                               bounds=bounds,
                               options={"maxiter": 200, "ftol": 1e-10})
                results_list.append(res)

            best_res = min(results_list, key=lambda r: r.fun)
            x_opt = best_res.x

            if fix_por and fix_de:
                opt_params = [por, x_opt[0], x_opt[1]*1e-5, x_opt[2]*1e-7, de_v]
            else:
                opt_params = [x_opt[0], x_opt[1], x_opt[2]*1e-5, x_opt[3]*1e-7, x_opt[4]*1e-10]

            opt_pred  = predict(*opt_params)
            achieved  = opt_pred[t_idx]

        st.success(f"Optimisation complete — target: {target_yield*100:.1f}%, achieved: {achieved*100:.2f}%")

        r1, r2, r3 = st.columns(3)
        if fix_por and fix_de:
            r1.metric("Optimal C₀",  f"{opt_params[1]:.3f} mol/m³")
            r2.metric("Optimal u₀",  f"{opt_params[2]*1e5:.2f} ×10⁻⁵ m/s")
            r3.metric("Optimal k_r", f"{opt_params[3]*1e7:.2f} ×10⁻⁷ m/s")
        else:
            r1.metric("Optimal por", f"{opt_params[0]:.3f}")
            r2.metric("Optimal C₀",  f"{opt_params[1]:.3f} mol/m³")
            r3.metric("Optimal u₀",  f"{opt_params[2]*1e5:.2f} ×10⁻⁵ m/s")

        fig_inv, ax_inv = plt.subplots(figsize=(8, 4))
        ax_inv.plot(TIME_DAYS, pred_A*100,   "o--", color="#888", lw=1.8,
                    ms=5, label="Current (Scenario A)", alpha=0.7)
        ax_inv.plot(TIME_DAYS, opt_pred*100, "o-",  color="#1D9E75", lw=2.5,
                    ms=7, label="Optimised parameters")
        ax_inv.axhline(target_yield*100, color="#D85A30", lw=1.5,
                       linestyle="--", label=f"Target: {target_yield*100:.1f}%")
        ax_inv.scatter([TIME_DAYS[t_idx]], [achieved*100],
                       s=120, zorder=5, color="#D85A30",
                       label=f"Achieved at {target_day}: {achieved*100:.2f}%")
        ax_inv.set_xlabel("Time (days)"); ax_inv.set_ylabel("X yield (%)")
        ax_inv.set_title("Optimised vs current parameter set")
        ax_inv.legend(fontsize=9); ax_inv.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig_inv); plt.close()


# ══════════════════════════════════════════════════════
# TAB 5 — DATASET EXPLORER
# ══════════════════════════════════════════════════════
with tab5:
    st.markdown("### COMSOL simulation dataset explorer")

    if df_data is not None:
        st.caption(f"{len(df_data)} simulations · 5 input parameters · 8 output time points")

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            x_axis = st.selectbox("X axis", FEATURES + ["X_60d"], index=1, key="x_ax")
        with c_exp2:
            y_axis = st.selectbox("Y axis", ["X_7d","X_14d","X_21d","X_30d","X_45d","X_60d"],
                                  index=5, key="y_ax")

        fig_exp, ax_exp = plt.subplots(figsize=(8, 4.5))
        sc = ax_exp.scatter(df_data[x_axis], df_data[y_axis],
                            c=df_data["k_r"], cmap="plasma",
                            alpha=0.7, s=35, edgecolors="none")
        plt.colorbar(sc, ax=ax_exp, label="k_r (colour)")
        ax_exp.set_xlabel(x_axis); ax_exp.set_ylabel(y_axis)
        ax_exp.set_title(f"{y_axis} vs {x_axis} — 200 COMSOL simulations (colour = k_r)")
        ax_exp.grid(True, alpha=0.2)
        # Mark current scenario A
        ax_exp.scatter([locals().get(x_axis, 0)], [pred_A[TIME_LABELS.index(y_axis)] if y_axis in TIME_LABELS else pred_A[7]],
                       s=150, color="red", zorder=5, marker="*",
                       label="Current scenario A")
        ax_exp.legend()
        plt.tight_layout()
        st.pyplot(fig_exp); plt.close()

        st.divider()
        st.markdown("**Correlation matrix — all parameters and outputs**")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
        corr_cols = FEATURES + ["X_7d","X_30d","X_60d"]
        corr_mat  = df_data[corr_cols].corr()
        im_corr   = ax_corr.imshow(corr_mat, cmap="coolwarm",
                                   vmin=-1, vmax=1, aspect="auto")
        ax_corr.set_xticks(range(len(corr_cols)))
        ax_corr.set_yticks(range(len(corr_cols)))
        ax_corr.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=9)
        ax_corr.set_yticklabels(corr_cols, fontsize=9)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                ax_corr.text(j, i, f"{corr_mat.iloc[i,j]:.2f}",
                             ha="center", va="center", fontsize=8,
                             color="white" if abs(corr_mat.iloc[i,j]) > 0.5 else "black")
        plt.colorbar(im_corr, ax=ax_corr)
        ax_corr.set_title("Pearson correlation: inputs vs outputs")
        plt.tight_layout()
        st.pyplot(fig_corr); plt.close()

        st.divider()
        st.markdown("**Distribution of X_60d across all 200 simulations**")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
        ax_hist.hist(df_data["X_60d"]*100, bins=25,
                     color="#378ADD", edgecolor="white", alpha=0.85)
        ax_hist.axvline(pred_A[7]*100, color="#D85A30", lw=2,
                        linestyle="--", label=f"Current: {pred_A[7]*100:.1f}%")
        pct = (df_data["X_60d"] < pred_A[7]).mean() * 100
        ax_hist.set_xlabel("X_60d (%)"); ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"X_60d distribution — current scenario is better than {pct:.0f}% of all simulations")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        st.pyplot(fig_hist); plt.close()

        st.divider()
        st.markdown("**Raw dataset (searchable)**")
        st.dataframe(df_data.round(6), use_container_width=True)

    else:
        st.warning("Dataset file not found. Place comsol_results_fixed.csv in the project folder.")


# ══════════════════════════════════════════════════════
# TAB 6 — EXPORT REPORT
# ══════════════════════════════════════════════════════
with tab6:
    st.markdown("### Export prediction report")
    st.caption("Generate a structured summary of the current Scenario A prediction")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        analyst_name = st.text_input("Analyst name", "Dnyaneshwari Deore")
        project_name = st.text_input("Project / ore type", "Heap Leach Column Study")
    with col_r2:
        notes = st.text_area("Additional notes", "Default parameter set. ANN surrogate model.")

    if st.button("📄 Generate report", type="primary"):
        lines = []
        lines.append("=" * 60)
        lines.append("HEAP LEACH EXTRACTION YIELD — PREDICTION REPORT")
        lines.append("=" * 60)
        lines.append(f"Analyst    : {analyst_name}")
        lines.append(f"Project    : {project_name}")
        lines.append(f"Model      : ANN surrogate (R² = 0.9902, RMSE = 0.0048)")
        lines.append(f"Notes      : {notes}")
        lines.append("")
        lines.append("INPUT PARAMETERS")
        lines.append("-" * 40)
        lines.append(f"  Porosity (por)       : {por:.3f}")
        lines.append(f"  Acid conc (C0)       : {C0:.3f} mol/m³")
        lines.append(f"  Darcy velocity (u0)  : {u0_v:.2e} m/s")
        lines.append(f"  Reaction rate (k_r)  : {kr_v:.2e} m/s")
        lines.append(f"  Diffusivity (D_eff)  : {de_v:.2e} m²/s")
        lines.append("")
        lines.append("PREDICTED EXTRACTION YIELD")
        lines.append("-" * 40)
        for t, y in zip(TIME_DAYS, pred_A):
            lines.append(f"  Day {t:>3}  :  {y*100:.3f}%  {badge(y)}")
        lines.append("")
        lines.append("KEY FINDINGS")
        lines.append("-" * 40)
        lines.append(f"  Final yield (60d)    : {pred_A[7]*100:.2f}%")
        peak_r = np.diff(pred_A*100) / np.diff(TIME_DAYS)
        pk_i   = np.argmax(peak_r)
        lines.append(f"  Peak leaching rate   : {peak_r[pk_i]:.4f}%/day "
                     f"(days {TIME_DAYS[pk_i]}–{TIME_DAYS[pk_i+1]})")
        t10 = next((TIME_DAYS[i] for i,v in enumerate(pred_A) if v>=0.10), None)
        lines.append(f"  Days to 10% yield    : {'Day ' + str(t10) if t10 else 'Not reached'}")
        lines.append("=" * 60)
        lines.append("Generated by: Heap Leach ML Predictor")
        lines.append("Physics-ML hybrid workflow: COMSOL + ANN")
        lines.append("=" * 60)

        report_text = "\n".join(lines)
        st.text(report_text)
        st.download_button(
            label="⬇️ Download report (.txt)",
            data=report_text,
            file_name=f"heap_leach_report_{project_name.replace(' ','_')}.txt",
            mime="text/plain"
        )

        # CSV export
        csv_data = pd.DataFrame({
            "Time (days)": TIME_DAYS,
            "X_yield":     pred_A,
            "Yield (%)":   pred_A * 100
        })
        st.download_button(
            label="⬇️ Download yield data (.csv)",
            data=csv_data.to_csv(index=False),
            file_name="prediction_yield_data.csv",
            mime="text/csv"
        )

st.divider()
st.caption("2nd Year MME · Physics-ML Hybrid Workflow · COMSOL Multiphysics + Python ANN · R² = 0.9902")
