import pandas as pd
import numpy as np

df = pd.read_csv("outputs/vehicle_log.csv")

danger_events = []
interaction_rows = []

TTC_THRESHOLD = 3.0
LATERAL_THRESHOLD = 2.5

for t, frame in df.groupby("time"):

    vehicles = frame.to_dict("records")

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):

            v1, v2 = vehicles[i], vehicles[j]

            dx = v1["x"] - v2["x"]
            dy = v1["y"] - v2["y"]

            dist = np.sqrt(dx**2 + dy**2)
            if dist > 50 or dist == 0:
                continue

            dvx = v1["vx"] - v2["vx"]
            dvy = v1["vy"] - v2["vy"]

            dot_product = (dx * dvx) + (dy * dvy)

            if dot_product < 0:

                v_rel_sq = (dvx**2) + (dvy**2)

                if v_rel_sq > 0.1:

                    ttc = -dot_product / v_rel_sq

                    lat_dist = abs(dx * dvy - dy * dvx) / np.sqrt(v_rel_sq)

                    rel_speed = np.sqrt(v_rel_sq)

                    drac = (rel_speed**2) / (2 * dist)

                    label = 0

                    if 0 < ttc < TTC_THRESHOLD and lat_dist < LATERAL_THRESHOLD:
                        label = 1

                        danger_events.append({
                            "time": t,
                            "ttc": ttc,
                            "drac": drac,
                            "lat_dist": lat_dist,
                            "v_rel": rel_speed
                        })

                    interaction_rows.append({
                        "time": int(t),
                        "distance": float(dist),
                        "relative_speed": float(rel_speed),
                        "lat_dist": float(lat_dist),
                        "ttc": float(ttc),
                        "drac": float(drac),
                        "label": label
                    })

# Convert to DataFrames
danger_df = pd.DataFrame(danger_events)
dataset_df = pd.DataFrame(interaction_rows)

# Save ML dataset
dataset_df.to_csv("outputs/interaction_dataset.csv", index=False)

print("Saved ML dataset to outputs/interaction_dataset.csv")
print("Dataset size:", len(dataset_df))

# --- Summary Statistics ---
print("\n=== SAFETY CONFLICT REPORT ===")
print(f"Total Critical Events: {len(danger_df)}")

if not danger_df.empty:

    print(f"Average TTC:           {danger_df['ttc'].mean():.2f}s")
    print(f"Minimum TTC:           {danger_df['ttc'].min():.2f}s")
    print(f"Average DRAC:          {danger_df['drac'].mean():.2f} m/s²")
    print(f"Max DRAC (Peak Risk):  {danger_df['drac'].max():.2f} m/s²")

    high_risk = danger_df[danger_df["ttc"] < 1.5].shape[0]

    print(f"High-Risk (TTC < 1.5s): {high_risk} events")

    print("\nSeverity Breakdown (DRAC):")
    print(f"- Low (0-3.35):       {danger_df[danger_df['drac'] <= 3.35].shape[0]}")
    print(f"- Emergency (> 3.35): {danger_df[danger_df['drac'] > 3.35].shape[0]}")