import os
import matplotlib.pyplot as plt
import pandas as pd


def draw_graph(logs, name, yscale):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for log in logs:
        path = os.path.join("logs", log)
        df = pd.read_csv(path, sep="\s+", header=None, names=["Time", log])
        ax.plot(df["Time"], df[log], label=log)
    ax.legend()
    ax.set_yscale(yscale)
    fig.tight_layout()
    fig.savefig(os.path.join("logs", name))
    print(f"Saved {logs} -> {name}")


logs = ["Ux_0", "UxFinalRes_0"]
draw_graph(logs, "Ux.png", "log")

logs = ["Uy_0", "UyFinalRes_0"]
draw_graph(logs, "Uy.png", "log")

# logs = ["Uz_0", "UzFinalRes_0"]
# draw_graph(logs, "Uz.png", "log")

logs = ["p_0", "p_0"]
draw_graph(logs, "p.png", "log")

logs = ["k_0", "kFinalRes_0"]
draw_graph(logs, "k.png", "log")

logs = ["epsilon_0", "epsilonFinalRes_0"]
draw_graph(logs, "epsilon.png", "log")

logs = ["epsAvg_0", "epsMin_0", "epsMax_0"]
draw_graph(logs, "epsMinMax.png", "log")


def calc_outlet_velocity_deviation(mean_file):
    path = os.path.join("postProcessing", mean_file)
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # Parse: "1    (-4.639561e-09 8.405160e-06 1.354001e-02)"
            parts = line.split("\t")
            time = float(parts[0].strip())
            # Extract values from parentheses
            vec_str = parts[1].strip().strip("()")
            ux, uy, uz = [float(v) for v in vec_str.split()]
            data.append([time, ux, uy, uz])
    df = pd.DataFrame(data, columns=["Time", "Ux", "Uy", "Uz"])

    Umean_ref = df.tail(30).mean()
    Umean_last = df.tail(1)

    eps = (Umean_last - Umean_ref) / Umean_ref
    print("\nOutlet velocity deviation:")
    print(f"Ux: abs value/deviation([m/s]/[-]) = {Umean_last['Ux'].iloc[0]: .2e}/{eps['Ux'].iloc[0]: .2e}")
    print(f"Uy: abs value/deviation([m/s]/[-]) = {Umean_last['Uy'].iloc[0]: .2e}/{eps['Uy'].iloc[0]: .2e}")
    print(f"Uz: abs value/deviation([m/s]/[-]) = {Umean_last['Uz'].iloc[0]: .2e}/{eps['Uz'].iloc[0]: .2e}")


calc_outlet_velocity_deviation("meanOutlet/0/surfaceFieldValue.dat")
