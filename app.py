import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG & CSS
# =========================================================
st.set_page_config(
    page_title="PG-RSE | FRP Shear Capacity Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        padding: 25px;
        border-radius: 14px;
        color: white;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.6);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .metric-subtitle { font-size: 0.75rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1. LOAD EXACT TRAINED MODEL PACKAGE
# =========================================================
@st.cache_resource
def load_model_package():
    model_file = 'trained_frp_model.pkl'
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return None

model_pkg = load_model_package()

# =========================================================
# 2. PHYSICS (ACI 440.2R) & AI PREDICTION FUNCTIONS
# =========================================================
def calculate_aci_vf(fc, tf, Ef_GPa, efu, wf, sf, hfe, alpha_deg, rm):
    try:
        Ef_MPa = Ef_GPa * 1000.0
        alpha_rad = np.radians(alpha_deg)
        Afv = 2.0 * tf * wf
        Le = 23300.0 / ((tf * Ef_MPa)**0.58)
        k1 = (fc / 27.0)**(2.0 / 3.0)
        
        if rm == 1:
            k2 = (hfe - Le) / hfe if hfe > Le else 0.0
        elif rm == 2:
            k2 = (hfe - 2.0 * Le) / hfe if hfe > (2.0 * Le) else 0.0
        else:
            k2 = 1.0 
            
        Kv = min(max((k1 * k2 * Le) / (11900.0 * efu), 0.0), 0.75) if rm in [1, 2] else 0.75
        efe = min(Kv * efu, 0.004)
        ffe = efe * Ef_MPa
        
        if sf > 0:
            Vf_N = (Afv * ffe * (np.sin(alpha_rad) + np.cos(alpha_rad)) * hfe) / sf
            return max(Vf_N / 1000.0, 0.0)
        return 0.0
    except Exception:
        return 0.0

def predict_shear_with_exact_ai(vf_code, lam, rho_sv, hfe, rho_f, alpha, rm, shape):
    if model_pkg is None:
        return 0.0, 0.0, [0.0, 0.0, 0.0]
    
    input_df = pd.DataFrame([{
        'λ': lam,
        'ρ_sv': rho_sv,
        'h_fe': hfe,
        'ρ_f': rho_f,
        'α': alpha,
        'V_f,code': vf_code,
        'R. M.': rm,
        'shape': 1 if shape.lower() == 't-beam' else 0
    }])
    
    model_preds = []
    weighted_preds = []
    
    for model, weight in zip(model_pkg['models'], model_pkg['weights']):
        pred_log = model.predict(input_df)[0]
        pred_raw = float(np.expm1(pred_log))
        model_preds.append(pred_raw)
        weighted_preds.append(weight * pred_raw)
        
    final_vf = float(sum(weighted_preds))
    residual = float(final_vf - vf_code)
    return final_vf, residual, model_preds

# =========================================================
# HEADER BANNER
# =========================================================
r2_score_disp = f"R² = {model_pkg['metrics']['r2']:.4f}" if model_pkg else "No Model Found"

st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚡ Physics-Guided ML: FRP Shear Predictor (PG-RSE)
            </h1>
            <p style="margin-top: 6px; margin-bottom: 0; color: #cbd5e1; font-size: 0.95rem;">
                Trained on 315 Experimental Tests via <b>Optuna Optimization & PSO Stacking</b>
            </p>
        </div>
        <div style="margin-top: 8px;">
            <span style="background: rgba(99, 102, 241, 0.25); border: 1px solid #6366f1; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #a5b4fc; font-weight: 700;">
                Model Accuracy: {r2_score_disp}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if model_pkg is None:
    st.error("⚠️ `trained_frp_model.pkl` not found! Run `python train.py` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "🎯 Real-Time Specimen Predictor",
    "📈 Dynamic Parametric Sweep",
    "📂 Batch Evaluation (.xlsx / .csv)"
])

# =========================================================
# TAB 1: REAL-TIME PREDICTOR
# =========================================================
with tab1:
    col_in, col_out = st.columns([1, 1.15], gap="large")
    
    with col_in:
        st.markdown("### 🛠️ Input Parameters")
        with st.expander("📐 Beam Geometry & Concrete", expanded=True):
            cg1, cg2 = st.columns(2)
            with cg1:
                lam = st.slider("Shear Span Ratio (λ = a/d)", 0.7, 5.0, 2.2, 0.1)
                hfe = st.slider("Effective FRP Depth (h_fe, mm)", 100.0, 700.0, 264.0, 5.0)
                fc = st.slider("Concrete Strength (f'_c, MPa)", 14.0, 75.0, 31.44, 1.0)
            with cg2:
                rho_sv = st.slider("Stirrup Ratio (ρ_sv, %)", 0.0, 0.8, 0.22, 0.01)
                shape = st.radio("Beam Cross-Section", ["Rectangular", "T-beam"], horizontal=True)
                rm_choice = st.selectbox("Wrapping Scheme (R.M.)", [
                    "1: U-wrap (3 Sides)",
                    "2: Side-Bonded (2 Sides)",
                    "3: Full Wrap (Closed Loop)"
                ])
                rm = int(rm_choice.split(":")[0])

        with st.expander("🧵 FRP Material & Strips", expanded=True):
            cf1, cf2 = st.columns(2)
            with cf1:
                tf = st.number_input("FRP Thickness (t_f, mm)", 0.05, 5.0, 0.111, 0.01, format="%.3f")
                Ef = st.number_input("Elastic Modulus (E_f, GPa)", 100.0, 300.0, 235.0, 5.0)
                efu = st.number_input("Ultimate Strain (ε_fu)", 0.005, 0.025, 0.0151, 0.001, format="%.4f")
            with cf2:
                wf = st.number_input("Strip Width (w_f, mm)", 1.0, 300.0, 62.0, 5.0)
                sf = st.number_input("Strip Spacing (s_f, mm)", 1.0, 500.0, 157.0, 5.0)
                alpha = st.slider("Fiber Angle (α, deg)", 45, 90, 90, 5)
                rho_f = st.number_input("FRP Ratio (ρ_f, %)", 0.02, 5.0, 0.058, 0.01)

    with col_out:
        st.markdown("### 🔮 Prediction Outputs")
        
        vf_code = calculate_aci_vf(fc, tf, Ef, efu, wf, sf, hfe, alpha, rm)
        vf_pred, residual, base_preds = predict_shear_with_exact_ai(vf_code, lam, rho_sv, hfe, rho_f, alpha, rm, shape)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">ACI 440.2R Code</div>
                <div class="metric-value" style="color: #38bdf8;">{vf_code:.2f} <span style="font-size:1rem;">kN</span></div>
                <div class="metric-subtitle">Physics Baseline</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            res_color = "#4ade80" if residual >= 0 else "#f87171"
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">Learned AI Residual</div>
                <div class="metric-value" style="color: {res_color};">{residual:+.2f} <span style="font-size:1rem;">kN</span></div>
                <div class="metric-subtitle">ΔV_f Correction</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">Final PG-RSE Output</div>
                <div class="metric-value" style="color: #c084fc;">{vf_pred:.2f} <span style="font-size:1rem;">kN</span></div>
                <div class="metric-subtitle">PSO Ensemble Model</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        
        # Sub-Model Contributions
        with st.expander("🔍 Ensemble Component Breakdown", expanded=False):
            w = model_pkg['weights']
            st.write(f"**XGBoost (Weight: {w[0]*100:.1f}%):** {base_preds[0]:.2f} kN")
            st.write(f"**LightGBM (Weight: {w[1]*100:.1f}%):** {base_preds[1]:.2f} kN")
            st.write(f"**CatBoost (Weight: {w[2]*100:.1f}%):** {base_preds[2]:.2f} kN")

        # Interactive Gauge
        max_gauge = max(vf_pred, vf_code, 100.0) * 1.3
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=vf_pred,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': vf_code, 'increasing': {'color': "#4ade80"}, 'decreasing': {'color': "#f87171"}},
            number={'suffix': " kN", 'font': {'size': 32, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, max_gauge], 'tickwidth': 1, 'tickcolor': "#64748b"},
                'bar': {'color': "#818cf8", 'thickness': 0.3},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, vf_code], 'color': "rgba(56, 189, 248, 0.25)"},
                    {'range': [vf_code, max_gauge], 'color': "rgba(192, 132, 252, 0.15)"}
                ],
                'threshold': {'line': {'color': "#38bdf8", 'width': 3}, 'thickness': 0.8, 'value': vf_code}
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#f8fafc"},
            height=260,
            margin=dict(l=20, r=20, t=25, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# =========================================================
# TAB 2: PARAMETRIC SWEEP
# =========================================================
with tab2:
    st.markdown("### 📈 Live Parametric Sweep (ACI Code vs Trained AI)")
    
    p_col1, p_col2 = st.columns([1, 3])
    with p_col1:
        param_choice = st.selectbox("Select Parameter to Sweep", [
            "FRP Effective Depth (h_fe)",
            "Concrete Strength (f'_c)",
            "FRP Elastic Modulus (E_f)",
            "Shear Span Ratio (λ)"
        ])
        steps = st.slider("Resolution (data points)", 15, 60, 30)
    
    with p_col2:
        if "h_fe" in param_choice:
            sweep_vals = np.linspace(100.0, 680.0, steps)
            code_vals = [calculate_aci_vf(fc, tf, Ef, efu, wf, sf, h, alpha, rm) for h in sweep_vals]
            ai_vals = [predict_shear_with_exact_ai(c, lam, rho_sv, h, rho_f, alpha, rm, shape)[0] for c, h in zip(code_vals, sweep_vals)]
            x_label = "Effective Depth h_fe (mm)"
        elif "f'_c" in param_choice:
            sweep_vals = np.linspace(15.0, 75.0, steps)
            code_vals = [calculate_aci_vf(f, tf, Ef, efu, wf, sf, hfe, alpha, rm) for f in sweep_vals]
            ai_vals = [predict_shear_with_exact_ai(c, lam, rho_sv, hfe, rho_f, alpha, rm, shape)[0] for c in code_vals]
            x_label = "Concrete Compressive Strength f'_c (MPa)"
        elif "E_f" in param_choice:
            sweep_vals = np.linspace(100.0, 300.0, steps)
            code_vals = [calculate_aci_vf(fc, tf, e, efu, wf, sf, hfe, alpha, rm) for e in sweep_vals]
            ai_vals = [predict_shear_with_exact_ai(c, lam, rho_sv, hfe, rho_f, alpha, rm, shape)[0] for c in code_vals]
            x_label = "FRP Modulus E_f (GPa)"
        else:
            sweep_vals = np.linspace(0.8, 4.8, steps)
            code_vals = [vf_code] * steps
            ai_vals = [predict_shear_with_exact_ai(vf_code, l, rho_sv, hfe, rho_f, alpha, rm, shape)[0] for l in sweep_vals]
            x_label = "Shear Span Ratio λ (a/d)"
            
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_vals, y=code_vals, mode='lines', name='ACI 440.2R Code', line=dict(color='#38bdf8', width=3, dash='dash')))
        fig_sweep.add_trace(go.Scatter(x=sweep_vals, y=ai_vals, mode='lines+markers', name='Trained PG-RSE AI', line=dict(color='#c084fc', width=3.5)))
        
        fig_sweep.update_layout(
            title=f"Sensitivity Curve: $V_f$ vs {x_label}",
            xaxis_title=x_label,
            yaxis_title="Shear Contribution $V_f$ (kN)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

# =========================================================
# TAB 3: BATCH EVALUATION
# =========================================================
with tab3:
    st.markdown("### 📂 Bulk Specimen Processing")
    up_file = st.file_uploader("Upload Excel Specimen Dataset", type=['xlsx', 'xls', 'csv'])
    if up_file is not None:
        try:
            df_b = pd.read_excel(up_file) if (up_file.name.endswith('.xlsx') or up_file.name.endswith('.xls')) else pd.read_csv(up_file)
            st.success(f"Loaded {len(df_b)} specimens successfully.")
            st.dataframe(df_b.head(5))
        except Exception as e:
            st.error(f"Error reading file: {e}")