import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Iterable, Optional

def points_to_cells(points: np.ndarray, eps: float, d: int) -> np.ndarray:
    assert points.shape[1] == d, f"points's column number should be d={d}"
    P = np.clip(points, 0.0, 1.0 - 1e-12)
    k = int(np.ceil(1.0 / eps))
    idx = np.floor(P * k).astype(int)
    cell_ids = np.ravel_multi_index(idx.T, dims=(k,) * d)
    return cell_ids

def grid_coverage(points: np.ndarray, eps: float, d: int) -> Tuple[int, int, float]:
    k = int(np.ceil(1.0 / eps))
    total_cells = k ** d
    if points.size == 0:
        return 0, total_cells, 0.0
    cell_ids = points_to_cells(points, eps, d)
    hit = np.unique(cell_ids).size
    return hit, total_cells, hit / total_cells

def run_grid_coverage_experiment(
    d: int = 3,                      
    n_label: int = 200,              
    unlabeled_grid: Iterable[int] = (200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200),
    eps: float = 0.05,               
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(123)

    L = rng.uniform(0.0, 1.0, size=(n_label, d))
    U_max = max(unlabeled_grid)
    U_pool = rng.uniform(0.0, 1.0, size=(U_max, d))

    cells_L, total_cells, cov_L = grid_coverage(L, eps, d)

    records = []
    for n_u in unlabeled_grid:
        U = U_pool[:n_u]
        LU = np.vstack([L, U])
        cells_LU, total_cells2, cov_LU = grid_coverage(LU, eps, d)
        assert total_cells == total_cells2
        records.append({
            'n_labeled': n_label,
            "n_unlabeled": n_u,
            "coverage_label_only": cov_L,
            "coverage_combined": cov_LU,
            "cells_label_only": cells_L,
            "cells_combined": cells_LU,
            "total_cells": total_cells
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    d = 3
    n_label = 1600
    unlabeled_grid = (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200)
    eps = 0.05  

    df = run_grid_coverage_experiment(
        d=d,
        n_label=n_label,
        unlabeled_grid=unlabeled_grid,
        eps=eps,
    )
    print(df)

    plt.figure(figsize=(6, 4))
    plt.plot(df['n_unlabeled'] / df['n_labeled'], df["coverage_combined"], marker="o", label='Labeled & Unlabeled', linewidth=2, markersize=8)
    plt.axhline(y=df["coverage_label_only"].iloc[0], 
                color='r', linestyle='--', label='Labeled Only', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Ratio', fontsize=12)
    plt.ylabel('Empirical coverage', fontsize=12)
    plt.legend()
    plt.tight_layout()
    save_path = "coverage_increase.pdf"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
