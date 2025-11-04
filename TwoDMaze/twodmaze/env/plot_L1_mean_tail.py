import matplotlib.pyplot as plt
import csv

# load your policy_drift.csv
output_dir = "soccer_stat"
rows = []
with open(f"{output_dir}/policy_drift.csv", newline="") as f:
    r = csv.DictReader(f)
    for line in r:
        rows.append(line)

episodes   = [int(r["episode"]) for r in rows]
l1_A_mean  = [float(r["l1_A_mean"]) for r in rows]
l1_B_mean  = [float(r["l1_B_mean"]) for r in rows]

# A zoomed
plt.figure()
plt.plot(episodes, l1_A_mean)
plt.title("Policy Drift — L1 mean (A) (zoomed)")
plt.xlabel("Episode")
plt.ylabel("L1(A)")
plt.ylim(0.0, 0.12)  # adjust if your tail is smaller/larger
plt.tight_layout()
plt.savefig(f"{output_dir}/policy_drift_l1_mean_A_zoom.png")
plt.close()

# B zoomed
plt.figure()
plt.plot(episodes, l1_B_mean)
plt.title("Policy Drift — L1 mean (B) (zoomed)")
plt.xlabel("Episode")
plt.ylabel("L1(B)")
plt.ylim(0.0, 0.12)
plt.tight_layout()
plt.savefig(f"{output_dir}/policy_drift_l1_mean_B_zoom.png")
plt.close()
