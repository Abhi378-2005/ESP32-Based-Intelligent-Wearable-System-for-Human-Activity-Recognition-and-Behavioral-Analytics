import re
import math
import time
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

ACTIVITY_CLASSES = [
    "Sitting (Ergonomic)",
    "Slouching (Sustained Tilt)",
    "Walking",
    "Running / Workout",
    "Desk Reach / Micro-Stretch",
    "Fall / High Impact"
]

ACTIVITY_META = {
    "Sitting (Ergonomic)": {"emoji": "🧘‍♂️", "color": "#00E676", "kcal_per_min": 1.3, "xp_rate": 5, "badge": "Zen Spine"},
    "Slouching (Sustained Tilt)": {"emoji": "🦐", "color": "#FF5252", "kcal_per_min": 1.2, "xp_rate": -3, "badge": "Shrimp Alert!"},
    "Walking": {"emoji": "🚶‍♂️", "color": "#00B0FF", "kcal_per_min": 4.2, "xp_rate": 8, "badge": "Active Strider"},
    "Running / Workout": {"emoji": "🏃‍♂️", "color": "#FF9100", "kcal_per_min": 9.8, "xp_rate": 15, "badge": "Cardio Beast"},
    "Desk Reach / Micro-Stretch": {"emoji": "🙆‍♂️", "color": "#FFD740", "kcal_per_min": 2.0, "xp_rate": 4, "badge": "Dynamic Mover"},
    "Fall / High Impact": {"emoji": "🚨", "color": "#FF1744", "kcal_per_min": 1.5, "xp_rate": 0, "badge": "Impact Detected"}
}


def parse_esp32_serial_line(line: str):
    if not line or "Rel Pitch:" not in line:
        return None

    pitch_m = re.search(r"Rel Pitch:\s*([-+]?\d*\.?\d+)", line)
    roll_m = re.search(r"Rel Roll:\s*([-+]?\d*\.?\d+)", line)
    svm_m = re.search(r"SVM:\s*([-+]?\d*\.?\d+)", line)
    gyro_m = re.search(r"Gyro:\s*([-+]?\d*\.?\d+)", line)
    status_m = re.search(r"STATUS:\s*(\[.*?\])", line)

    if not pitch_m or not roll_m:
        return None

    pitch = float(pitch_m.group(1))
    roll = float(roll_m.group(1))
    status_str = status_m.group(1) if status_m else "[OK - Good Posture]"

    if svm_m and gyro_m:
        svm = float(svm_m.group(1))
        gyro = float(gyro_m.group(1))
    else:
        tilt_mag = math.sqrt(pitch**2 + roll**2)
        svm = round(1.0 + 0.002 * tilt_mag + random.uniform(-0.02, 0.02), 2)
        gyro = round(tilt_mag * 0.65 + random.uniform(0.5, 4.5), 1)

    return {
        "pitch": pitch,
        "roll": roll,
        "svm": svm,
        "gyro": gyro,
        "status": status_str
    }


def classify_live_sample(pitch: float, roll: float, svm: float, gyro: float, status: str, pitch_std: float = 1.5):
    tilt_mag = math.sqrt(pitch**2 + roll**2)

    if svm >= 2.4 or svm <= 0.45:
        return "Fall / High Impact", 0.98
    if svm >= 1.55 or (gyro >= 115.0 and pitch_std >= 7.0):
        return "Running / Workout", 0.94
    if (1.15 <= svm < 1.55 and gyro >= 38.0) or (pitch_std >= 4.2 and gyro >= 32.0):
        return "Walking", 0.91
    if "ALERT" in status or (tilt_mag > 25.0 and gyro < 30.0 and "Tilted" not in status):
        return "Slouching (Sustained Tilt)", 0.96
    if "Tilted" in status or (tilt_mag > 22.0 and gyro >= 18.0):
        return "Desk Reach / Micro-Stretch", 0.89
    return "Sitting (Ergonomic)", 0.97


def simulate_sensor_tick(mode: str, step_idx: int):
    t = step_idx * 0.35
    if mode == "Auto Demo Tour (Cycles All Activities)":
        cycle = (step_idx // 16) % 5
        modes = [
            "Sitting (Ergonomic)",
            "Desk Reach / Micro-Stretch",
            "Slouching (Sustained Tilt)",
            "Walking",
            "Running / Workout"
        ]
        active_mode = modes[cycle]
    else:
        active_mode = mode

    if active_mode == "Sitting (Ergonomic)":
        pitch = 2.0 * math.sin(t * 0.5) + random.uniform(-1.2, 1.2)
        roll = 1.5 * math.cos(t * 0.4) + random.uniform(-1.0, 1.0)
        svm = 1.00 + random.uniform(-0.03, 0.03)
        gyro = 4.0 + random.uniform(-2.0, 3.0)
        status = "[OK - Good Posture]"
    elif active_mode == "Desk Reach / Micro-Stretch":
        pitch = 28.5 + 3.0 * math.sin(t) + random.uniform(-1.5, 1.5)
        roll = 10.0 + random.uniform(-2.0, 2.0)
        svm = 1.08 + random.uniform(-0.04, 0.05)
        gyro = 26.0 + random.uniform(-4.0, 6.0)
        rem = round(max(0.3, 2.8 - ((step_idx % 8) * 0.35)), 1)
        status = f"[Tilted... {rem}s until alert]"
    elif active_mode == "Slouching (Sustained Tilt)":
        pitch = 34.0 + 1.8 * math.sin(t * 0.3) + random.uniform(-1.0, 1.0)
        roll = -14.0 + random.uniform(-1.5, 1.5)
        svm = 1.01 + random.uniform(-0.02, 0.02)
        gyro = 5.5 + random.uniform(-2.0, 2.0)
        status = "[ALERT - BAD POSTURE!]"
    elif active_mode == "Walking":
        pitch = 8.0 + 7.5 * math.sin(t * 3.2) + random.uniform(-2.0, 2.0)
        roll = 5.5 * math.cos(t * 3.2) + random.uniform(-2.0, 2.0)
        svm = 1.28 + 0.14 * abs(math.sin(t * 3.2)) + random.uniform(-0.03, 0.03)
        gyro = 58.0 + 18.0 * math.sin(t * 3.2) + random.uniform(-5.0, 5.0)
        status = "[OK - Active Motion]"
    elif active_mode == "Running / Workout":
        pitch = 14.0 + 14.0 * math.sin(t * 6.0) + random.uniform(-3.5, 3.5)
        roll = 11.0 * math.cos(t * 6.0) + random.uniform(-3.0, 3.0)
        svm = 1.85 + 0.35 * abs(math.sin(t * 6.0)) + random.uniform(-0.05, 0.08)
        gyro = 145.0 + 35.0 * abs(math.cos(t * 6.0)) + random.uniform(-10.0, 10.0)
        status = "[OK - High Activity]"
    else:
        pitch = 62.0 + random.uniform(-8.0, 8.0)
        roll = -48.0 + random.uniform(-8.0, 8.0)
        svm = 2.85 + random.uniform(-0.15, 0.35)
        gyro = 240.0 + random.uniform(-20.0, 30.0)
        status = "[ALERT - IMPACT DETECTED!]"

    return {
        "pitch": round(pitch, 1),
        "roll": round(roll, 1),
        "svm": round(svm, 2),
        "gyro": round(gyro, 1),
        "status": status
    }


def train_random_forest_har(n_samples_per_class: int = 180):
    rng = np.random.default_rng(42)
    rows = []

    for label in ACTIVITY_CLASSES:
        for _ in range(n_samples_per_class):
            if label == "Sitting (Ergonomic)":
                mean_pitch = rng.normal(2.0, 4.5)
                mean_roll = rng.normal(0.0, 4.0)
                pitch_std = rng.uniform(0.3, 2.2)
                svm_rms = rng.normal(1.00, 0.03)
                gyro_energy = rng.normal(5.0, 2.5)
                zero_cross = rng.integers(0, 3)
            elif label == "Slouching (Sustained Tilt)":
                mean_pitch = rng.normal(33.0, 5.5)
                mean_roll = rng.normal(-12.0, 6.0)
                pitch_std = rng.uniform(0.5, 2.8)
                svm_rms = rng.normal(1.01, 0.03)
                gyro_energy = rng.normal(6.5, 3.0)
                zero_cross = rng.integers(0, 3)
            elif label == "Desk Reach / Micro-Stretch":
                mean_pitch = rng.normal(27.0, 5.0)
                mean_roll = rng.normal(14.0, 5.0)
                pitch_std = rng.uniform(3.5, 7.5)
                svm_rms = rng.normal(1.09, 0.05)
                gyro_energy = rng.normal(28.0, 7.0)
                zero_cross = rng.integers(2, 6)
            elif label == "Walking":
                mean_pitch = rng.normal(9.0, 4.0)
                mean_roll = rng.normal(3.0, 4.0)
                pitch_std = rng.uniform(5.0, 9.5)
                svm_rms = rng.normal(1.32, 0.08)
                gyro_energy = rng.normal(62.0, 12.0)
                zero_cross = rng.integers(5, 11)
            elif label == "Running / Workout":
                mean_pitch = rng.normal(15.0, 5.5)
                mean_roll = rng.normal(5.0, 5.5)
                pitch_std = rng.uniform(10.0, 18.0)
                svm_rms = rng.normal(1.92, 0.16)
                gyro_energy = rng.normal(155.0, 22.0)
                zero_cross = rng.integers(10, 18)
            else:
                mean_pitch = rng.normal(55.0, 12.0)
                mean_roll = rng.normal(-42.0, 12.0)
                pitch_std = rng.uniform(18.0, 32.0)
                svm_rms = rng.normal(2.75, 0.25)
                gyro_energy = rng.normal(235.0, 30.0)
                zero_cross = rng.integers(1, 5)

            rows.append({
                "Mean_Pitch_Deg": round(float(mean_pitch), 2),
                "Mean_Roll_Deg": round(float(mean_roll), 2),
                "Pitch_Std_Dev": round(float(pitch_std), 2),
                "SVM_RMS_g": round(float(svm_rms), 3),
                "Gyro_Energy_DPS": round(float(gyro_energy), 2),
                "Zero_Crossings": int(zero_cross),
                "Activity": label
            })

    df = pd.DataFrame(rows)
    feature_cols = ["Mean_Pitch_Deg", "Mean_Roll_Deg", "Pitch_Std_Dev", "SVM_RMS_g", "Gyro_Energy_DPS", "Zero_Crossings"]
    X = df[feature_cols]
    y = df["Activity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=ACTIVITY_CLASSES)

    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=True)

    return clf, df, acc, cm, importances, feature_cols