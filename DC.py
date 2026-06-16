#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# --------------------- Utility --------------------- #
def _floats(line: str):
    out = []
    for p in line.replace('D', 'E').split():
        try:
            out.append(float(p))
        except:
            pass
    return out

# --------------------- G96 Reader --------------------- #
def load_g96_all_frames(path: str) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
    coords_frames, box_frames = [], []
    cur_coords = []
    cur_box = None
    in_pos = False
    waiting_box = False
    box_vals = []

    def finalize_frame():
        nonlocal cur_coords, cur_box
        if len(cur_coords) == 0:
            return
        coords_frames.append(np.asarray(cur_coords, dtype=float))
        box_frames.append(None if cur_box is None else np.asarray(cur_box, dtype=float))
        cur_coords = []
        cur_box = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            up = line.upper()

            if up.startswith("POSITION") or up.startswith("POSITIONRED"):
                if len(cur_coords) > 0:
                    finalize_frame()
                in_pos = True
                continue
            if up.startswith("BOX"):
                waiting_box = True
                box_vals = []
                continue
            if up.startswith("END"):
                in_pos = False
                continue
            if waiting_box:
                vals = _floats(line)
                if vals:
                    box_vals += vals
                    if len(box_vals) >= 3:
                        if len(box_vals) >= 9:
                            mat = np.array(box_vals[:9], float).reshape(3, 3)
                            cur_box = np.linalg.norm(mat, axis=1)
                        else:
                            cur_box = np.array(box_vals[:3], float)
                        waiting_box = False
                continue
            if in_pos:
                vals = _floats(line)
                if len(vals) >= 3:
                    cur_coords.append(vals[-3:])

    if len(cur_coords) > 0:
        coords_frames.append(np.asarray(cur_coords, dtype=float))
        box_frames.append(None if cur_box is None else np.asarray(cur_box, dtype=float))

    if len(coords_frames) == 0:
        raise ValueError(f"No coordinates parsed from {path}")
    return coords_frames, box_frames

# --------------------- PBC helpers --------------------- #
def replicate_images(B: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
    if box is None:
        return B
    shifts = np.array([[i, j, k] for i in (-1, 0, 1)
                                 for j in (-1, 0, 1)
                                 for k in (-1, 0, 1)], dtype=float)
    return np.vstack([B + shift * box for shift in shifts])

# --------------------- Neighbor counting --------------------- #
def neighbor_counts_frame(A: np.ndarray, B: np.ndarray, box: Optional[np.ndarray], cutoff: float) -> np.ndarray:
    if B.size == 0:
        return np.zeros(len(A), dtype=int)
    Bimg = replicate_images(B, box)
    tree = cKDTree(Bimg)
    return np.array([len(tree.query_ball_point(a, cutoff)) for a in A], dtype=int)

# --------------------- RDF accumulate --------------------- #
def rdf_accumulate(A, B, box, bins, counts_acc, denom_acc):
    if B.size == 0 or A.size == 0:
        return
    Bimg = replicate_images(B, box)
    tree = cKDTree(Bimg)
    rmax = bins[-1]
    per_bin_counts = np.zeros(len(bins) - 1, dtype=float)

    for a in A:
        idxs = tree.query_ball_point(a, rmax)
        if not idxs:
            continue
        dists = np.linalg.norm(Bimg[idxs] - a, axis=1)
        hist, _ = np.histogram(dists, bins=bins)
        per_bin_counts += hist

    counts_acc += per_bin_counts
    V = np.prod(np.asarray(box, dtype=float))
    rhoB = len(B) / V
    shell_vol = (4.0 / 3.0) * math.pi * (bins[1:]**3 - bins[:-1]**3)
    denom_acc += len(A) * rhoB * shell_vol

# --------------------- First Solvation Shell Counter --------------------- #
def count_first_shell(A, B, box, rmin, rmax):
    """Count number of B atoms within first shell (rmin–rmax) for each A atom."""
    if B.size == 0:
        return np.zeros(len(A), dtype=int)
    Bimg = replicate_images(B, box)
    tree = cKDTree(Bimg)
    counts = np.zeros(len(A), dtype=int)
    for i, a in enumerate(A):
        idxs = tree.query_ball_point(a, rmax)
        if not idxs:
            continue
        dists = np.linalg.norm(Bimg[idxs] - a, axis=1)
        mask = (dists >= rmin) & (dists < rmax)
        counts[i] = np.count_nonzero(mask)
    return counts

# --------------------- CLI --------------------- #
def parse_args():
    p = argparse.ArgumentParser("RDF + Solvation Shell Probability Distribution + Neighbors")
    p.add_argument("--polymer", required=True)
    p.add_argument("--cations")
    p.add_argument("--anions")
    p.add_argument("--box", nargs=3, type=float, metavar=("LX", "LY", "LZ"))
    p.add_argument("--cutoff", type=float, default=0.6)
    p.add_argument("--rmax", type=float, default=2.0)
    p.add_argument("--dr", type=float, default=0.001)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()

# --------------------- MAIN --------------------- #
def main():
    a = parse_args()
    poly_frames, poly_boxes = load_g96_all_frames(a.polymer)
    cat_frames, ani_frames = [], []
    if a.cations:
        cat_frames, cat_boxes = load_g96_all_frames(a.cations)
    if a.anions:
        ani_frames, ani_boxes = load_g96_all_frames(a.anions)

    n_frames = len(poly_frames)
    if a.cations:
        n_frames = min(n_frames, len(cat_frames))
    if a.anions:
        n_frames = min(n_frames, len(ani_frames))
    print(f"Loaded {n_frames} frames for analysis")

    bins = np.arange(0.0, a.rmax + a.dr, a.dr)
    rmid = 0.5 * (bins[:-1] + bins[1:])
    rdf_counts = {"cations": np.zeros(len(bins)-1), "anions": np.zeros(len(bins)-1)}
    rdf_denom  = {"cations": np.zeros(len(bins)-1), "anions": np.zeros(len(bins)-1)}

    n_poly = len(poly_frames[0])
    neigh_sum = {"cations": np.zeros(n_poly), "anions": np.zeros(n_poly)}

    cation_counts_all = []
    anion_counts_all = []
    shell_cat = (0.0, 0.3)
    shell_ani = (0.0, 1.2)

    for k in range(n_frames):
        P = poly_frames[k]
        box = np.array(a.box) if a.box is not None else (poly_boxes[k] or [7.5,7.5,7.5])
        if a.cations:
            C = cat_frames[k]
            neigh_sum["cations"] += neighbor_counts_frame(P, C, box, a.cutoff)
            c_counts = count_first_shell(P, C, box, *shell_cat)
            cation_counts_all.extend(c_counts)
            rdf_accumulate(P, C, box, bins, rdf_counts["cations"], rdf_denom["cations"])
        if a.anions:
            A = ani_frames[k]
            neigh_sum["anions"] += neighbor_counts_frame(P, A, box, a.cutoff)
            a_counts = count_first_shell(P, A, box, *shell_ani)
            anion_counts_all.extend(a_counts)
            rdf_accumulate(P, A, box, bins, rdf_counts["anions"], rdf_denom["anions"])

    # ---- Save neighbor stats ----
    for lab in ("cations", "anions"):
        avg_counts = neigh_sum[lab] / n_frames
        np.savetxt(f"neighbors_polymer_{lab}.csv", np.column_stack([np.arange(len(avg_counts)), avg_counts]),
                   delimiter=",", header="index,count", comments="")
        print(f"[ok] neighbors ({lab}) -> mean={avg_counts.mean():.3f}")

    # ---- RDF plots ----
    for lab in ("cations", "anions"):
        g = np.divide(rdf_counts[lab], rdf_denom[lab], out=np.zeros_like(rdf_counts[lab]), where=rdf_denom[lab]>0)
        np.savetxt(f"rdf_polymer_{lab}.csv", np.column_stack([rmid, g]),
                   delimiter=",", header="r,g(r)", comments="")
        if a.plot:
            plt.figure(figsize=(10, 4), dpi=300)
            plt.plot(rmid, g, color="steelblue", linewidth=0.8, label="Raw RDF")
            plt.title(f"RDF polymer–{lab} (dr={a.dr})")
            plt.xlabel("r (nm)")
            plt.ylabel("g(r)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"rdf_polymer_{lab}.png", dpi=300)
            plt.close()
            print(f"[ok] rdf_polymer_{lab}.png saved")

    # ---- Probability Distribution P(n) ----
    plt.figure(figsize=(8,5), dpi=300)
    if len(cation_counts_all) > 0:
        bins_c = np.arange(0, max(cation_counts_all)+2) - 0.5
        hist_c, _ = np.histogram(cation_counts_all, bins=bins_c)
        prob_c = hist_c / hist_c.sum()
        plt.plot(np.arange(0, max(cation_counts_all)+1), prob_c, 'o-', color='blue', label='Cations (0–0.3 nm)')
    if len(anion_counts_all) > 0:
        bins_a = np.arange(0, max(anion_counts_all)+2) - 0.5
        hist_a, _ = np.histogram(anion_counts_all, bins=bins_a)
        prob_a = hist_a / hist_a.sum()
        plt.plot(np.arange(0, max(anion_counts_all)+1), prob_a, 'o-', color='orange', label='Anions (0–1.2 nm)')

    plt.xlabel("Number of ions n in first solvation shell")
    plt.ylabel("Probability P(n)")
    plt.title("Solvation Shell Population Distribution (Polymer–Ion)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("solvation_shell_probability_distribution.png", dpi=300)
    plt.close()
    print("[ok] solvation_shell_probability_distribution.png saved ")

if __name__ == "__main__":
    main()
