import numpy as np

for path in [
    "data/standardized/sn_pantheon.npz",
    "data/standardized/sn_pantheon_shoes.npz",
]:
    try:
        d = np.load(path, allow_pickle=True)
        print(f"\n[CHECK] {path}")
        print("keys:", list(d.keys()))
        if "z" in d: print("N_data =", len(d["z"]))
        if "cov" in d: print("cov shape =", d["cov"].shape)
        if "obs" in d: print("obs mean =", float(np.mean(d['obs'])))
        if "cov" in d: print("median σμ =", np.median(np.sqrt(np.diag(d['cov']))))
    except Exception as e:
        print(f"[ERROR] cannot load {path}: {e}")

