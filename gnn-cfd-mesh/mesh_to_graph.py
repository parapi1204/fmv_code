"""
CFDメッシュをグラフで表現する - 検証スクリプト
ブログ記事1「CFDメッシュをグラフで表現する｜ノード・エッジ・境界条件の設計」用

矩形流路 + 円柱まわりの2D非構造メッシュを自作し、
GNN(MeshGraphNet系)の入力となる3点セット
  - x          : ノード特徴 [N, F_n]
  - edge_index : 接続情報   [2, E]
  - edge_attr  : エッジ特徴 [E, F_e]
を numpy だけで組み立てて中身を確かめる。
(PyTorch Geometric に渡す直前のデータをそのまま numpy で作るイメージ)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# 1. 計算領域とメッシュ点の生成（矩形流路 + 円柱）
# ----------------------------------------------------------------------
LX, LY = 3.0, 1.0            # 流路の長さ・高さ
CX, CY, R = 0.8, 0.5, 0.15   # 円柱の中心と半径

def make_points():
    pts = []
    # 外周境界（流入・流出・上下壁）
    n_in = 21
    for y in np.linspace(0, LY, n_in):
        pts.append((0.0, y))          # inlet
        pts.append((LX, y))           # outlet
    for x in np.linspace(0, LX, 61)[1:-1]:
        pts.append((x, 0.0))          # bottom wall
        pts.append((x, LY))           # top wall
    # 円柱表面
    for th in np.linspace(0, 2 * np.pi, 40, endpoint=False):
        pts.append((CX + R * np.cos(th), CY + R * np.sin(th)))
    # 内部点（円柱の内側は除外、円柱近傍をやや密に）
    n_trial = 2000
    cand = rng.uniform([0.02, 0.02], [LX - 0.02, LY - 0.02], size=(n_trial, 2))
    d = np.hypot(cand[:, 0] - CX, cand[:, 1] - CY)
    keep = d > R * 1.15
    # 間引いて点数を調整（近すぎる点を除去する簡易ポアソン風フィルタ）
    cand = cand[keep]
    selected = []
    min_dist = 0.06
    for p in cand:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > min_dist for q in selected):
            selected.append(tuple(p))
    pts.extend(selected)
    return np.array(pts)

pos = make_points()                    # [N, 2] ノード座標
N = len(pos)

# ----------------------------------------------------------------------
# 2. Delaunay分割 → 円柱内部の三角形を除去 → エッジ抽出
# ----------------------------------------------------------------------
tri = Delaunay(pos)
centers = pos[tri.simplices].mean(axis=1)
inside_cyl = np.hypot(centers[:, 0] - CX, centers[:, 1] - CY) < R
simplices = tri.simplices[~inside_cyl]

# 三角形の3辺をエッジとして集める（重複除去 → 双方向化）
edges = set()
for a, b, c in simplices:
    for i, j in ((a, b), (b, c), (c, a)):
        edges.add((min(i, j), max(i, j)))
edges = np.array(sorted(edges))                       # [E_undirected, 2]
# GNNでは「i→j」「j→i」の両方向を持たせるのが基本（双方向 message passing）
edge_index = np.concatenate([edges, edges[:, ::-1]]).T   # [2, E]
E = edge_index.shape[1]

# ----------------------------------------------------------------------
# 3. ノード種別（境界条件）の割り当て → one-hot
# ----------------------------------------------------------------------
INTERIOR, INLET, OUTLET, WALL = 0, 1, 2, 3
node_type = np.full(N, INTERIOR)
eps = 1e-6
node_type[np.abs(pos[:, 0] - 0.0) < eps] = INLET
node_type[np.abs(pos[:, 0] - LX) < eps] = OUTLET
node_type[(np.abs(pos[:, 1]) < eps) | (np.abs(pos[:, 1] - LY) < eps)] = WALL
on_cyl = np.abs(np.hypot(pos[:, 0] - CX, pos[:, 1] - CY) - R) < 1e-3
node_type[on_cyl] = WALL

one_hot = np.eye(4)[node_type]                        # [N, 4]

# ----------------------------------------------------------------------
# 4. 物理量（初期条件の例）を作り、ノード特徴 x を組み立てる
#    ここでは「流入で放物形速度分布、壁でno-slip」という状態を模擬
# ----------------------------------------------------------------------
u = np.zeros(N)
v = np.zeros(N)
inlet_mask = node_type == INLET
u[inlet_mask] = 4.0 * pos[inlet_mask, 1] * (LY - pos[inlet_mask, 1]) / LY**2
u[node_type == INTERIOR] = 0.5                        # 適当な内部初期値
u[node_type == WALL] = 0.0                            # no-slip

# ノード特徴 = [u, v, one-hot(4)] → MeshGraphNet(渦放出例)と同じ6次元
x = np.column_stack([u, v, one_hot])                  # [N, 6]

# ----------------------------------------------------------------------
# 5. エッジ特徴 = [dx, dy, |d|]（相対位置ベクトルとその長さ）
#    絶対座標をノードに入れず相対位置をエッジに入れる → 並進不変性
# ----------------------------------------------------------------------
src, dst = edge_index
rel = pos[dst] - pos[src]                             # [E, 2]
dist = np.linalg.norm(rel, axis=1, keepdims=True)     # [E, 1]
edge_attr = np.hstack([rel, dist])                    # [E, 3]

# ----------------------------------------------------------------------
# 6. 中身の確認（記事に載せる出力）
# ----------------------------------------------------------------------
print(f"ノード数 N = {N}, エッジ数 E = {E}（双方向）")
print(f"x          : {x.shape}   (u, v, one-hot 4種)")
print(f"edge_index : {edge_index.shape}")
print(f"edge_attr  : {edge_attr.shape} (dx, dy, |d|)")
print("\n--- ノード0（inlet上の点）の特徴 ---")
i = np.where(inlet_mask)[0][10]
print(f"座標 {pos[i]}, x[i] = {np.round(x[i], 3)}")
print("\n--- エッジ0の特徴 ---")
print(f"{src[0]} -> {dst[0]}, edge_attr = {np.round(edge_attr[0], 3)}")
print(f"\nノード種別の内訳: interior={np.sum(node_type==0)}, "
      f"inlet={np.sum(node_type==1)}, outlet={np.sum(node_type==2)}, "
      f"wall={np.sum(node_type==3)}")

# ----------------------------------------------------------------------
# 7. 可視化（記事用の図）
# ----------------------------------------------------------------------
plt.rcParams["font.size"] = 11

# 図1: メッシュ = グラフ（ノード種別を色分け）
fig, ax = plt.subplots(figsize=(10, 4))
for a, b in edges:
    ax.plot(pos[[a, b], 0], pos[[a, b], 1], "-", color="0.8", lw=0.5, zorder=1)
colors = {INTERIOR: "#9e9e9e", INLET: "#1976d2", OUTLET: "#e64a19", WALL: "#2e7d32"}
labels = {INTERIOR: "interior", INLET: "inlet", OUTLET: "outlet", WALL: "wall"}
for t in (INTERIOR, INLET, OUTLET, WALL):
    m = node_type == t
    ax.scatter(pos[m, 0], pos[m, 1], s=14, c=colors[t], label=labels[t], zorder=2)
ax.set_aspect("equal")
ax.legend(loc="upper right", ncol=4, framealpha=0.9)
ax.set_title("CFD mesh as a graph: nodes colored by boundary type")
fig.tight_layout()
fig.savefig("fig1_mesh_as_graph.png", dpi=150)

# 図2: ノード特徴(u)の分布とエッジ特徴の説明
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
sc = axes[0].scatter(pos[:, 0], pos[:, 1], c=u, s=14, cmap="viridis")
axes[0].set_aspect("equal")
axes[0].set_title("Node feature: u (initial condition)")
plt.colorbar(sc, ax=axes[0], shrink=0.8)

# 1本のエッジを拡大して相対位置ベクトルを図示
k = np.argmax((node_type[src] == INTERIOR) & (dist[:, 0] > 0.05))
p1, p2 = pos[src[k]], pos[dst[k]]
axes[1].plot(*zip(p1, p2), "ko-", ms=8)
axes[1].annotate("", xy=p2, xytext=p1,
                 arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
mid = (p1 + p2) / 2
axes[1].text(mid[0], mid[1] + 0.01,
             f"edge_attr = (dx, dy, |d|)\n= ({rel[k,0]:.3f}, {rel[k,1]:.3f}, {dist[k,0]:.3f})",
             color="crimson", fontsize=11)
axes[1].set_title("Edge feature: relative position vector")
axes[1].set_aspect("equal")
pad = 0.08
axes[1].set_xlim(min(p1[0], p2[0]) - pad, max(p1[0], p2[0]) + pad)
axes[1].set_ylim(min(p1[1], p2[1]) - pad, max(p1[1], p2[1]) + pad)
fig.tight_layout()
fig.savefig("fig2_node_edge_features.png", dpi=150)
print("\n図を保存しました: fig1_mesh_as_graph.png, fig2_node_edge_features.png")
