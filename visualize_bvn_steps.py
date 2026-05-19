import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Sinkhorn scaling (to create a doubly stochastic matrix)
# ----------------------------------------------------------------------
def sinkhorn(A, max_iter=1000, tol=1e-6):
    A = np.array(A, dtype=float)
    A = A / A.sum()
    for _ in range(max_iter):
        A = A / A.sum(axis=1, keepdims=True)
        A = A / A.sum(axis=0, keepdims=True)
        if np.max(np.abs(A.sum(1)-1)) < tol and np.max(np.abs(A.sum(0)-1)) < tol:
            break
    return A

# ----------------------------------------------------------------------
# Graph drawing helper (exactly as you provided)
# ----------------------------------------------------------------------
def draw_bipartite(R, matching_edges=None, title="", ax=None, node_size=800):
    n = R.shape[0]
    G = nx.Graph()
    row_nodes = [f"r{i}" for i in range(n)]
    col_nodes = [f"c{j}" for j in range(n)]
    G.add_nodes_from(row_nodes, bipartite=0)
    G.add_nodes_from(col_nodes, bipartite=1)
    for i in range(n):
        for j in range(n):
            if R[i, j] > 1e-8:
                G.add_edge(row_nodes[i], col_nodes[j], weight=R[i, j])

    pos = {row_nodes[i]: (0, -i) for i in range(n)}
    pos.update({col_nodes[j]: (1, -j) for j in range(n)})

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    nx.draw_networkx_nodes(G, pos, nodelist=row_nodes, node_color='lightblue',
                           node_size=node_size, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=col_nodes, node_color='lightcoral',
                           node_size=node_size, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=18, ax=ax)

    edges_all = list(G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=edges_all, width=1.8,
                           edge_color='grey', alpha=0.6, ax=ax)

    if matching_edges is not None:
        me = [(f"r{i}", f"c{j}") for (i, j) in matching_edges]
        nx.draw_networkx_edges(G, pos, edgelist=me, width=3,
                               edge_color='red', alpha=0.9, ax=ax)

    edge_labels = {(f"r{i}", f"c{j}"): f"{R[i,j]:.2f}"
                   for i in range(n) for j in range(n) if R[i,j] > 1e-8}
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=12, label_pos=0.2, ax=ax)
    ax.set_title(title)
    ax.axis('off')

# ----------------------------------------------------------------------
# Main iterative visualization
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 3
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 14,
            "figure.figsize": (10, 5),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "lines.linewidth": 3,
            "lines.markersize": 6,
        }
    )
    # Create a random symmetric doubly stochastic matrix
    A = np.random.uniform(0.5, 1.5, (n, n))
    A = (A + A.T) / 2
    S = sinkhorn(A)

    precision = 0.21
    max_iter = 10   # safety limit
    residual = S.copy()
    total_weight = 0.0
    steps = []      # will store (matching_edges, weight, residual_before, residual_after)

    for it in range(max_iter):
        # Build bipartite graph from current residual
        G = nx.Graph()
        row_nodes = np.arange(n)
        col_nodes = np.arange(n) + n
        G.add_nodes_from(row_nodes, bipartite=0)
        G.add_nodes_from(col_nodes, bipartite=1)
        for i in range(n):
            for j in range(n):
                if residual[i, j] > 0:
                    G.add_edge(i, j + n, weight=residual[i, j])

        # Maximum‑weight perfect matching
        matching_set = nx.algorithms.matching.max_weight_matching(
            G, maxcardinality=True, weight="weight")
        if len(matching_set) != n:
            break

        # Convert matching to (row, col) pairs
        prow = np.zeros(n, dtype=int)
        for u, v in matching_set:
            if u < n:
                prow[u] = v - n
            else:
                prow[v] = u - n
        weight = np.min(residual[np.arange(n), prow])
        matching_edges = [(i, prow[i]) for i in range(n)]

        # Store snapshot before subtraction
        residual_before = residual.copy()

        # Subtract weighted permutation
        residual[np.arange(n), prow] -= weight
        total_weight += weight

        steps.append((matching_edges, weight, residual_before, residual.copy()))

        # Check stopping criterion
        if np.linalg.norm(residual, 1) <= precision:
            break

    # ---------- Plotting: one row per iteration, two columns ----------
    num_steps = len(steps)
    fig, axes = plt.subplots(num_steps, 2, figsize=(9, 3.5 * num_steps))
    # Ensure axes is 2D even for a single row
    if num_steps == 1:
        axes = axes.reshape(1, -1)

    for row, (match, w, before, after) in enumerate(steps):
        draw_bipartite(before, matching_edges=match,
                       title=f"Iteration {row+1}: Max‑weight matching (weight = {w:.3f})",
                       ax=axes[row, 0])
        draw_bipartite(after, matching_edges=None,
                       title=f"Residual after subtraction",
                       ax=axes[row, 1])

    plt.tight_layout(pad=2.0)
    plt.savefig("bvn_iterative_steps.pdf", bbox_inches='tight', dpi=150)
    plt.savefig("bvn_iterative_steps.png", bbox_inches='tight', dpi=150)
    plt.show()