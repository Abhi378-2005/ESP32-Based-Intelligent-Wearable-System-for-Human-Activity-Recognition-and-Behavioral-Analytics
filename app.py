import time
import math
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False

from utils.activity_engine import (
    ACTIVITY_CLASSES,
    ACTIVITY_META,
    parse_esp32_serial_line,
    classify_live_sample,
    simulate_sensor_tick,
    train_random_forest_har
)

st.set_page_config(
    page_title="SpineSync AI • PostureGuard Pro HAR Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top right, #111927 0%, #0B0F19 100%);
        color: #F0F4F8;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0F2C59 0%, #00695C 55%, #004D40 100%);
        border: 1px solid rgba(0, 230, 118, 0.35);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 18px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #FFFFFF;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        color: #B2DFDB;
        margin-top: 4px;
    }
    .glass-card {
        background: rgba(22, 31, 48, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #90A4AE;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #FFFFFF;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #80CBC4;
        margin-top: 2px;
    }
    .activity-pill {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.4px;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    seed_rows = []
    now_ts = time.time() - 30
    for i in range(30):
        sim = simulate_sensor_tick("Auto Demo Tour (Cycles All Activities)", i)
        act, conf = classify_live_sample(sim["pitch"], sim["roll"], sim["svm"], sim["gyro"], sim["status"])
        seed_rows.append({
            "timestamp": datetime.fromtimestamp(now_ts + i).strftime("%H:%M:%S"),
            "step": i,
            "pitch": sim["pitch"],
            "roll": sim["roll"],
            "svm": sim["svm"],
            "gyro": sim["gyro"],
            "status": sim["status"],
            "activity": act,
            "confidence": conf
        })
    st.session_state.history = seed_rows
    st.session_state.step_counter = 30
    st.session_state.xp_score = 420
    st.session_state.streak_ticks = 14
    st.session_state.slouch_alerts = 2
    st.session_state.calories_burned = 18.4
    st.session_state.serial_conn = None

@st.cache_resource
def load_ml_pipeline():
    return train_random_forest_har(n_samples_per_class=180)

rf_model, synthetic_df, rf_acc, rf_cm, rf_importances, feature_cols = load_ml_pipeline()

with st.sidebar:
    st.markdown("## ⚡ SpineSync Control Deck")
    st.caption("ESP32 + MPU6050 Wearable HAR & Posture Hub")

    source_mode = st.radio(
        "📡 Telemetry Input Mode",
        [
            "🚀 Interactive Simulator (Demo)",
            "🔌 Live USB Serial Cable",
            "🔵 Wireless Bluetooth SPP (PostureGuard_Pro)"
        ],
        index=0
    )

    if source_mode == "🚀 Interactive Simulator (Demo)":
        sim_scenario = st.selectbox(
            "🎭 Select Activity Scenario",
            [
                "Auto Demo Tour (Cycles All Activities)",
                "Sitting (Ergonomic)",
                "Desk Reach / Micro-Stretch",
                "Slouching (Sustained Tilt)",
                "Walking",
                "Running / Workout",
                "Fall / High Impact"
            ],
            index=0
        )
        live_stream = st.toggle("▶️ Live Auto-Stream (Real-Time)", value=False)
        stream_speed = st.slider("⏱️ Stream Refresh Speed (sec)", 0.3, 2.0, 0.7, 0.1)
        if st.button("⚡ Step +1 Sensor Reading", use_container_width=True):
            st.session_state.trigger_single_step = True

    else:  # Both USB Serial and Bluetooth SPP use Virtual COM Ports via pyserial
        live_stream = False
        stream_speed = 0.4
        if not PYSERIAL_AVAILABLE:
            st.error("`pyserial` not installed. Run `pip install pyserial`.")
        else:
            port_objs = list(serial.tools.list_ports.comports())
            port_labels = [f"{p.device} — {p.description}" for p in port_objs]
            port_map = {f"{p.device} — {p.description}": p.device for p in port_objs}

            if source_mode == "🔵 Wireless Bluetooth SPP (PostureGuard_Pro)":
                st.info("Pair **PostureGuard_Pro** in Windows Bluetooth Settings, then pick its Outgoing COM port below.")

            if port_labels:
                chosen_label = st.selectbox("Select COM Port", port_labels)
                selected_port = port_map[chosen_label]
            else:
                selected_port = st.selectbox("Select COM Port (Manual)", ["COM3", "COM4", "COM5", "COM6", "COM7"])

            baud_rate = st.selectbox("Baud Rate", [115200, 9600], index=0)
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("🔗 Connect", use_container_width=True):
                    try:
                        if st.session_state.serial_conn and st.session_state.serial_conn.is_open:
                            st.session_state.serial_conn.close()
                        st.session_state.serial_conn = serial.Serial(selected_port, baud_rate, timeout=0.5)
                        st.success(f"Connected to {selected_port}!")
                    except Exception as e:
                        st.error(f"Port error: {e}")
            with col_c2:
                if st.button("❌ Disconnect", use_container_width=True):
                    if st.session_state.serial_conn:
                        st.session_state.serial_conn.close()
                        st.session_state.serial_conn = None
                    st.info("Disconnected.")
            live_stream = st.toggle("▶️ Stream Live Telemetry", value=False)

    st.divider()
    st.markdown("### 🎛️ Edge Thresholds (Mirror ESP32)")
    custom_tilt_limit = st.slider("Tilt Alert Threshold (± deg)", 15.0, 45.0, 25.0, 1.0)
    debounce_sec = st.slider("Slouch Debounce Timer (sec)", 1.0, 5.0, 3.0, 0.5)

    st.divider()
    st.markdown("### 🏆 Wearable Player Profile")
    level = max(1, st.session_state.xp_score // 150 + 1)
    st.metric("Player Level", f"Lvl {level} • Spine Guardian", f"{st.session_state.xp_score} XP")
    st.progress(min(1.0, (st.session_state.xp_score % 150) / 150.0), text=f"{st.session_state.xp_score % 150}/150 XP to Level {level+1}")

    if st.button("🔄 Reset Session Telemetry", use_container_width=True):
        st.session_state.pop("history", None)
        st.rerun()

def append_telemetry_sample(sample_dict):
    st.session_state.step_counter += 1
    recent_pitches = [r["pitch"] for r in st.session_state.history[-6:]] + [sample_dict["pitch"]]
    pitch_std = float(np.std(recent_pitches)) if len(recent_pitches) > 1 else 1.2

    act, conf = classify_live_sample(
        sample_dict["pitch"],
        sample_dict["roll"],
        sample_dict["svm"],
        sample_dict["gyro"],
        sample_dict["status"],
        pitch_std=pitch_std
    )

    meta = ACTIVITY_META[act]
    st.session_state.xp_score = max(0, st.session_state.xp_score + meta["xp_rate"])
    st.session_state.calories_burned = round(st.session_state.calories_burned + (meta["kcal_per_min"] / 60.0), 2)

    if act == "Slouching (Sustained Tilt)":
        st.session_state.streak_ticks = 0
        st.session_state.slouch_alerts += 1
    else:
        st.session_state.streak_ticks += 1

    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "step": st.session_state.step_counter,
        "pitch": sample_dict["pitch"],
        "roll": sample_dict["roll"],
        "svm": sample_dict["svm"],
        "gyro": sample_dict["gyro"],
        "status": sample_dict["status"],
        "activity": act,
        "confidence": conf
    })
    if len(st.session_state.history) > 120:
        st.session_state.history = st.session_state.history[-120:]

if getattr(st.session_state, "trigger_single_step", False) or live_stream:
    st.session_state.trigger_single_step = False
    if source_mode == "🚀 Interactive Simulator (Demo)":
        sim_data = simulate_sensor_tick(sim_scenario, st.session_state.step_counter)
        append_telemetry_sample(sim_data)
    elif st.session_state.serial_conn and st.session_state.serial_conn.is_open:
        try:
            raw_line = st.session_state.serial_conn.readline().decode("utf-8", errors="ignore").strip()
            parsed = parse_esp32_serial_line(raw_line)
            if parsed:
                append_telemetry_sample(parsed)
        except Exception:
            pass

df_hist = pd.DataFrame(st.session_state.history)
latest = df_hist.iloc[-1]
cur_act = latest["activity"]
cur_meta = ACTIVITY_META[cur_act]

good_ratio = len(df_hist[df_hist["activity"] != "Slouching (Sustained Tilt)"]) / max(1, len(df_hist))
wellness_score = int(round(good_ratio * 100))

st.markdown(f"""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
        <div>
            <p class="hero-title">⚡ SPINESYNC AI • POSTUREGUARD™ PRO</p>
            <p class="hero-subtitle">Real-Time 6-Axis IMU Activity Recognition, 3D Kinematic Orientation & Behavioral Wellness Engine</p>
        </div>
        <div style="text-align:right;">
            <span style="background:{cur_meta['color']}22; border:2px solid {cur_meta['color']}; color:{cur_meta['color']};" class="activity-pill">
                {cur_meta['emoji']} {cur_act} ({int(latest['confidence']*100)}% Conf)
            </span>
            <div style="font-size:0.8rem; color:#B2DFDB; margin-top:6px;">
                ESP32 Status: <code>{latest['status']}</code>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="kpi-label">🎯 Detected Activity</div>
        <div class="kpi-value" style="font-size:1.3rem; color:{cur_meta['color']};">{cur_meta['emoji']} {cur_act.split(' ')[0]}</div>
        <div class="kpi-sub">Badge: {cur_meta['badge']}</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="kpi-label">📐 Relative Pitch / Roll</div>
        <div class="kpi-value">{latest['pitch']:+.1f}° / {latest['roll']:+.1f}°</div>
        <div class="kpi-sub">Threshold: ±{custom_tilt_limit:.0f}° ({debounce_sec:.0f}s Debounce)</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="kpi-label">💚 Spine Wellness Score</div>
        <div class="kpi-value" style="color:{'#00E676' if wellness_score>=80 else '#FFD740' if wellness_score>=60 else '#FF5252'};">{wellness_score}%</div>
        <div class="kpi-sub">Streak: {st.session_state.streak_ticks} clean ticks</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="glass-card">
        <div class="kpi-label">🔥 Active Energy & G-Load</div>
        <div class="kpi-value">{st.session_state.calories_burned:.1f} kcal</div>
        <div class="kpi-sub">SVM: {latest['svm']:.2f} g | Gyro: {latest['gyro']:.0f}°/s</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class="glass-card">
        <div class="kpi-label">🦐 Slouch Alerts Blocked</div>
        <div class="kpi-value" style="color:#FF8A80;">{st.session_state.slouch_alerts} Alerts</div>
        <div class="kpi-sub">3s Debounce Active</div>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Telemetry & 3D Tilt Radar",
    "🏃‍♂️ Activity Analytics & Behavioral Insights",
    "🧠 Machine Learning HAR Lab (Random Forest)",
    "🔧 Hardware Architecture & Serial Console"
])

with tab1:
    col_left, col_right = st.columns([1.15, 1.85])
    with col_left:
        st.markdown("#### 🎯 2D Pitch vs. Roll Bullseye Target")
        theta = np.linspace(0, 2 * np.pi, 100)
        fig_polar = go.Figure()
        fig_polar.add_trace(go.Scatter(
            x=custom_tilt_limit * np.cos(theta),
            y=custom_tilt_limit * np.sin(theta),
            mode="lines", fill="toself", fillcolor="rgba(0, 230, 118, 0.08)",
            line=dict(color="#00E676", width=2, dash="dash"),
            name=f"Safe Zone (±{custom_tilt_limit:.0f}°)"
        ))
        trail_df = df_hist.tail(12)
        fig_polar.add_trace(go.Scatter(
            x=trail_df["roll"], y=trail_df["pitch"],
            mode="lines+markers", line=dict(color="rgba(0, 176, 255, 0.45)", width=2),
            marker=dict(size=6, color="rgba(0, 176, 255, 0.5)"), name="Recent Motion Trail"
        ))
        fig_polar.add_trace(go.Scatter(
            x=[latest["roll"]], y=[latest["pitch"]],
            mode="markers+text", marker=dict(size=18, color=cur_meta["color"], line=dict(color="#FFFFFF", width=2)),
            text=[f"  {cur_meta['emoji']}"], textposition="top right", name="Current Posture Vector"
        ))
        fig_polar.update_layout(
            height=360, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17, 25, 39, 0.7)",
            xaxis=dict(title="Relative Roll (° Left / Right)", range=[-55, 55], zeroline=True, zerolinecolor="#546E7A", gridcolor="#1E293B"),
            yaxis=dict(title="Relative Pitch (° Forward / Back)", range=[-55, 55], zeroline=True, zerolinecolor="#546E7A", gridcolor="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="#E2E8F0")
        )
        st.plotly_chart(fig_polar, use_container_width=True)

    with col_right:
        st.markdown("#### 📈 Real-Time Kinematic Angle Stream (Pitch & Roll)")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["pitch"], mode="lines+markers", name="Rel Pitch (°)", line=dict(color="#00E676", width=2.5), marker=dict(size=5)))
        fig_ts.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["roll"], mode="lines+markers", name="Rel Roll (°)", line=dict(color="#00B0FF", width=2.5), marker=dict(size=5)))
        fig_ts.add_hline(y=custom_tilt_limit, line_dash="dot", line_color="#FF5252", annotation_text=f"+{custom_tilt_limit:.0f}° Alert Limit")
        fig_ts.add_hline(y=-custom_tilt_limit, line_dash="dot", line_color="#FF5252", annotation_text=f"-{custom_tilt_limit:.0f}° Alert Limit")
        fig_ts.update_layout(
            height=360, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17, 25, 39, 0.7)",
            xaxis=dict(title="Timestamp", gridcolor="#1E293B"),
            yaxis=dict(title="Angle Deviation (Degrees)", range=[-60, 60], gridcolor="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="#E2E8F0")
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("#### ⚡ Dynamic Motion Intensity: Total G-Force (SVM) & Gyro Angular Velocity")
    fig_dyn = go.Figure()
    fig_dyn.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["svm"], mode="lines", name="Accel Magnitude SVM (g)", line=dict(color="#FFD740", width=2.5), fill="tozeroy", fillcolor="rgba(255, 215, 64, 0.08)"))
    fig_dyn.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["gyro"] / 50.0, mode="lines", name="Gyro Intensity (scaled /50 dps)", line=dict(color="#E040FB", width=2, dash="dot")))
    fig_dyn.update_layout(
        height=230, margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17, 25, 39, 0.7)",
        xaxis=dict(gridcolor="#1E293B"), yaxis=dict(title="G-Force / Scaled Gyro", gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#E2E8F0")
    )
    st.plotly_chart(fig_dyn, use_container_width=True)

with tab2:
    c_a1, c_a2 = st.columns([1.1, 1.4])
    with c_a1:
        st.markdown("#### 🍩 Session Activity Distribution")
        act_counts = df_hist["activity"].value_counts().reset_index()
        act_counts.columns = ["Activity", "Count"]
        color_map = {k: v["color"] for k, v in ACTIVITY_META.items()}
        fig_pie = px.pie(act_counts, names="Activity", values="Count", hole=0.52, color="Activity", color_discrete_map=color_map)
        fig_pie.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_a2:
        st.markdown("#### 🕸️ Multi-Axis Behavioral Biomechanics Radar")
        posture_stability = max(10, min(100, int(100 - np.std(df_hist["pitch"]) * 2.2)))
        ergonomic_align = wellness_score
        dynamic_mobility = min(100, int(np.mean(df_hist["gyro"]) * 1.1 + 25))
        debounce_resilience = min(100, int(85 + (len(df_hist[df_hist["activity"] == "Desk Reach / Micro-Stretch"]) * 2)))
        endurance_score = min(100, int(60 + st.session_state.streak_ticks * 1.5))
        categories = ["Posture Stability", "Ergonomic Alignment", "Dynamic Mobility", "Debounce Filtering", "Spine Endurance"]
        values = [posture_stability, ergonomic_align, dynamic_mobility, debounce_resilience, endurance_score]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill="toself", fillcolor="rgba(0, 230, 118, 0.22)", line=dict(color="#00E676", width=2.5), name="Current User Profile"))
        fig_radar.update_layout(height=340, polar=dict(bgcolor="rgba(17, 25, 39, 0.7)", radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155"), angularaxis=dict(gridcolor="#334155")), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"), margin=dict(l=40, r=40, t=30, b=30))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("#### 📋 Recent Activity & Posture Event Log")
    display_df = df_hist[["timestamp", "activity", "pitch", "roll", "svm", "gyro", "status"]].tail(15).iloc[::-1]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export Full Session Telemetry (CSV)", data=csv_bytes, file_name=f"PostureGuard_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

with tab3:
    st.markdown("#### 🧠 Random Forest Human Activity Recognition (HAR) Pipeline")
    st.write(f"Trained on **1,080 sliding-window IMU feature vectors** across 6 movement and posture classes. Validation Accuracy: **`{rf_acc*100:.2f}%`**")
    m_col1, m_col2 = st.columns([1.2, 1.1])
    with m_col1:
        st.markdown("##### 📊 Confusion Matrix (6 Activity Classes)")
        fig_cm = px.imshow(rf_cm, x=[c.split(" ")[0] for c in ACTIVITY_CLASSES], y=[c.split(" ")[0] for c in ACTIVITY_CLASSES], text_auto=True, color_continuous_scale="Tealgrn", labels=dict(x="Predicted Activity", y="True Activity", color="Windows"))
        fig_cm.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)
    with m_col2:
        st.markdown("##### 🔍 Random Forest Feature Importance")
        fig_imp = px.bar(rf_importances, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Mint")
        fig_imp.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17, 25, 39, 0.7)", font=dict(color="#E2E8F0"), margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.markdown("##### 🧪 Interactive ML Inference Sandbox — Test Custom Sensor Features")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    test_p = s1.number_input("Mean Pitch (°)", -60.0, 60.0, float(latest["pitch"]), 1.0)
    test_r = s2.number_input("Mean Roll (°)", -60.0, 60.0, float(latest["roll"]), 1.0)
    test_std = s3.number_input("Pitch Std Dev", 0.1, 35.0, 2.0, 0.5)
    test_svm = s4.number_input("SVM RMS (g)", 0.3, 3.5, float(latest["svm"]), 0.05)
    test_gyro = s5.number_input("Gyro Energy (°/s)", 0.0, 300.0, float(latest["gyro"]), 5.0)
    test_zc = s6.number_input("Zero Crossings", 0, 25, 2, 1)

    input_vec = pd.DataFrame([{
        "Mean_Pitch_Deg": test_p,
        "Mean_Roll_Deg": test_r,
        "Pitch_Std_Dev": test_std,
        "SVM_RMS_g": test_svm,
        "Gyro_Energy_DPS": test_gyro,
        "Zero_Crossings": test_zc
    }])
    pred_class = rf_model.predict(input_vec)[0]
    pred_probs = rf_model.predict_proba(input_vec)[0]
    p_meta = ACTIVITY_META[pred_class]
    st.success(f"**Random Forest Prediction:** {p_meta['emoji']} **{pred_class}** (Max Probability: `{max(pred_probs)*100:.1f}%`)")

with tab4:
    h1, h2 = st.columns([1.1, 1.2])
    with h1:
        st.markdown("#### 🔌 Custom Milled Single-Sided PCB Pinout")
        st.markdown("""
        | ESP32 Pin | MPU6050 Pin | Signal | Electrical Function |
        | :--- | :--- | :--- | :--- |
        | **3V3** | **VCC** | Power (`3.3V`) | Regulated 3.3V Logic Supply |
        | **GND** | **GND** | Ground (`0V`) | Common Reference Ground |
        | **GPIO 21** | **SDA** | I²C Data | Bi-directional Serial Data (`0x68`) |
        | **GPIO 22** | **SCL** | I²C Clock | Synchronous Clock Line |
        """)
        st.info(
            "💡 **Dual USB + Bluetooth SPP Architecture:** The ESP32 transmits identical telemetry strings "
            "simultaneously over the physical USB UART (`Serial`) and Wireless Bluetooth SPP (`SerialBT` "
            "named **PostureGuard_Pro**), preventing any 2.4GHz Wi-Fi radio interference."
        )
    with h2:
        st.markdown("#### 💻 Live Raw Telemetry Monitor (`115200 baud`)")
        serial_lines = [
            f"[{r['timestamp']}] Rel Pitch: {r['pitch']:+.1f} deg | Rel Roll: {r['roll']:+.1f} deg | SVM: {r['svm']:.2f} g | Gyro: {r['gyro']:.1f} dps | STATUS: {r['status']}  --> [{r['activity']}]"
            for r in st.session_state.history[-12:]
        ]
        st.code("\n".join(serial_lines), language="bash")

if live_stream:
    time.sleep(stream_speed)
    st.rerun()