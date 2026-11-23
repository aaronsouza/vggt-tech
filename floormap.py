"""
3D -> 2D floor map from VGGT outputs.

Inputs  (produced by run_vggt_room.py):
    <in_dir>/room.ply      colored point cloud in VGGT world coordinates
    <in_dir>/cameras.npz   extrinsic (S,3,4, world->camera), intrinsic (S,3,3), paths

Outputs (in <out_dir>):
    floormap.png           top-down colored map (floor = z 0, up = +z)
    floormap_walls.png     top-down occupancy of points above the floor (walls/furniture)
    homographies.npz       per camera: H_img_to_floor (original-image pixels -> floor x,y)
    floor_aligned.ply      point cloud with the floor leveled (z = height above floor)

Usage:
    python floormap.py [in_dir] [out_dir] [--scale S] [--res 1000]
      --scale S : real units per VGGT unit (e.g. meters). Get it by measuring one known
                  distance: S = real_length / length_in_model_units. Default 1 (arbitrary units).
"""
import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image


def load_ply(path):
    with open(path) as f:
        n = 0
        for line in f:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.startswith("end_header"):
                break
        a = np.loadtxt(f, max_rows=n)
    return a[:, :3], a[:, 3:6].astype(np.uint8)


def estimate_up(R):
    # OpenCV camera: +y points down. In world coords the camera "down" is row 1 of R.
    up = -np.mean(R[:, 1, :], axis=0)
    return up / np.linalg.norm(up)


def fit_floor(P, up, iters=3000, thr_frac=0.01, min_cos=0.9, seed=0, max_pts=200000):
    rng = np.random.default_rng(seed)
    if len(P) > max_pts:
        Q = P[rng.choice(len(P), max_pts, replace=False)]
    else:
        Q = P
    scale = np.median(np.linalg.norm(Q - np.median(Q, axis=0), axis=1))
    thr = thr_frac * scale
    best_n, best_d, best_cnt = None, None, -1
    for _ in range(iters):
        s = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if abs(n @ up) < min_cos:      # only (roughly) horizontal planes
            continue
        d = -n @ s[0]
        cnt = int(np.sum(np.abs(Q @ n + d) < thr))
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    if best_n is None:
        raise RuntimeError("No horizontal plane found. Is the floor visible in the photos?")
    inl = np.abs(Q @ best_n + best_d) < thr
    pts = Q[inl]
    c = pts.mean(axis=0)
    n = np.linalg.svd(pts - c, full_matrices=False)[2][-1]   # least-squares refinement
    if n @ up < 0:
        n = -n
    return n, c, inl.mean(), thr


def align_world(R, t, n, c, x_hint):
    """Return Q (3x3) and new extrinsics so that floor is z=0 and +z is up.
    X' = Q (X - c);  R' = R Q^T;  t' = R c + t."""
    z = n / np.linalg.norm(n)
    x = x_hint - (x_hint @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Q = np.stack([x, y, z], axis=0)
    R2 = np.einsum("sij,kj->sik", R, Q)           # R Q^T
    t2 = np.einsum("sij,j->si", R, c) + t         # R c + t
    return Q, R2, t2


def floor_homography(K, R, t, s=1.0):
    """floor plane (X,Y,0) in real units -> image pixels. s = real units per model unit."""
    return K @ np.column_stack([R[:, 0], R[:, 1], s * t])


def K_to_original(K, W, H):
    # VGGT's K is for its resized image; principal point is the image center.
    sx = W / (2.0 * K[0, 2])
    sy = H / (2.0 * K[1, 2])
    return np.diag([sx, sy, 1.0]) @ K


def rasterize(xy, C, order_key, lo, sc, W, H, bg=255):
    img = np.full((H, W, 3), bg, np.uint8)
    ij = np.floor((xy - lo) * sc).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    ij, Cc, key = ij[ok], C[ok], order_key[ok]
    o = np.argsort(key)                           # low first, high points overwrite
    ij, Cc = ij[o], Cc[o]
    img[H - 1 - ij[:, 1], ij[:, 0]] = Cc
    return img


def write_ply(path, P, C):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(P))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(f, np.hstack([P, C]), fmt="%.4f %.4f %.4f %d %d %d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", nargs="?", default="~/amc-test/room_out")
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--res", type=int, default=1000)
    a = ap.parse_args()

    in_dir = os.path.expanduser(a.in_dir)
    out_dir = os.path.expanduser(a.out_dir or os.path.join(in_dir, "floor"))
    os.makedirs(out_dir, exist_ok=True)

    P, C = load_ply(os.path.join(in_dir, "room.ply"))
    cam = np.load(os.path.join(in_dir, "cameras.npz"), allow_pickle=True)
    E, K, paths = cam["extrinsic"], cam["intrinsic"], [str(p) for p in cam["paths"]]
    R, t = E[:, :, :3], E[:, :, 3]
    print(f"{len(P)} points, {len(R)} cameras", flush=True)

    up = estimate_up(R)
    n, c, frac, thr = fit_floor(P, up)
    print(f"floor plane: {frac*100:.1f}% of sampled points are inliers", flush=True)

    x_hint = R[0][0, :]                           # camera-0 right direction in world coords
    Q, R2, t2 = align_world(R, t, n, c, x_hint)

    # camera heights above floor (sanity check: should be positive and similar to each other)
    centers = -np.einsum("sji,sj->si", R2, t2)    # -R'^T t'
    h = centers[:, 2] * a.scale
    print("camera heights above floor (units x scale):", np.round(h, 3))
    if np.mean(centers[:, 2]) < 0:
        print("WARNING: cameras are below the fitted plane - it is probably the ceiling, not the floor.")

    # aligned point cloud, scaled
    Pn = (P - c) @ Q.T * a.scale
    write_ply(os.path.join(out_dir, "floor_aligned.ply"), Pn, C)

    # keep points near/above the floor, drop far outliers
    keep = (Pn[:, 2] > -2 * thr * a.scale)
    Pk, Ck = Pn[keep], C[keep]
    lo = np.percentile(Pk[:, :2], 1, axis=0)
    hi = np.percentile(Pk[:, :2], 99, axis=0)
    sc = a.res / float(max(hi - lo))
    W = int((hi[0] - lo[0]) * sc) + 1
    H = int((hi[1] - lo[1]) * sc) + 1
    print(f"map: {W}x{H} px, {1/sc:.4f} units per pixel ({sc:.1f} px per unit)", flush=True)

    topdown = rasterize(Pk[:, :2], Ck, Pk[:, 2], lo, sc, W, H)
    cv2.imwrite(os.path.join(out_dir, "floormap.png"), cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR))

    # occupancy of things above the floor (walls / furniture): points between 0.15 and 1.8 x camera height
    cam_h = float(np.mean(h)) if np.mean(h) > 0 else 1.0
    band = (Pk[:, 2] > 0.1 * cam_h) & (Pk[:, 2] < 1.8 * cam_h)
    occ = np.zeros((H, W), np.uint8)
    ij = np.floor((Pk[band, :2] - lo) * sc).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    ij = ij[ok]
    np.add.at(occ, (H - 1 - ij[:, 1], ij[:, 0]), 1)
    occ = (np.clip(occ, 0, 5) * 51).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "floormap_walls.png"), 255 - occ)

    # per-camera homographies: original image pixels -> floor coordinates (units x scale)
    Hs = []
    for i, p in enumerate(paths):
        w0, h0 = Image.open(p).size
        Ko = K_to_original(K[i], w0, h0)
        # floor coords are in real units (model * scale): [r1 r2 s*t] handles the scaling
        H_f2i = floor_homography(Ko, R2[i], t2[i], s=a.scale)
        Hs.append(np.linalg.inv(H_f2i))
    np.savez(os.path.join(out_dir, "homographies.npz"),
             H_img_to_floor=np.array(Hs), paths=np.array(paths),
             origin_lo=lo, px_per_unit=sc, scale=a.scale)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()