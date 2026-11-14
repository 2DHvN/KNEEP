#!/usr/bin/env python3
"""Standalone single-condition LRAMB experiment.

Port of demos/lramb_single_run.ipynb; no notebook or IPython needed at runtime.
Default: 32x32 Poisson g=0.3, ell=1, D=0.1, radius=8; radial BVP +
periodic continuation; 100000 burn steps; 128/16/1 ensemble members.
Euler-Maruyama, per-step midpoint medium EP, width 128 / hidden depth 2.

Examples (from KNEEP):
    python -u scripts/lramb_single_run.py --device cuda:0
    python -u scripts/lramb_single_run.py --device cuda:0 --resume
    python -u scripts/lramb_single_run.py --stage evaluate --device cuda:0
    python -u scripts/lramb_single_run.py --config single_run.json

Edit the dictionaries in run(), or override them using --config JSON.
This is one experiment on one selected device, not a multi-GPU sweep.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--lramb-root", type=Path, help="Sibling LRAMB repo by default.")
    parser.add_argument("--output-dir", type=Path, help="Data root; same default as notebook.")
    parser.add_argument("--fit-output-dir", type=Path, help="Optional separate fit/checkpoint root.")
    parser.add_argument("--config", type=Path, help="JSON overrides for GRID, BASE, DATA, MODEL, TRAIN, BVP, ROOT.")
    parser.add_argument("--stage", choices=("all", "generate", "evaluate"), default="all",
                        help="all: generate/reuse, fit, evaluate; generate: data/profile only; "
                             "evaluate: require cached data and checkpoint, no fitting.")
    parser.add_argument("--resume", action="store_true",
                        help="Restore the matching last.pt optimizer/sampler/history; finish configured iterations.")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best",
                        help="Checkpoint used for held-out evaluation.")
    parser.add_argument("--movie", choices=("auto", "mp4", "html", "none"), default="auto",
                        help="auto: MP4 if ffmpeg exists, otherwise standalone HTML.")
    parser.add_argument("--movie-max-frames", type=int, default=180)
    parser.add_argument("--movie-fps", type=int, default=12)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.resume and args.stage != "all":
        parser.error("--resume is used with --stage all.")
    if min(args.cpu_threads, args.movie_max_frames, args.movie_fps) < 1:
        parser.error("Thread/frame/FPS counts must be positive.")
    return args


def run(args):
    # ---- Imports and repository paths ----
    from pathlib import Path
    import hashlib
    import json
    import math
    import os
    import shutil
    import sys
    import tempfile
    import time
    import numpy as np
    from scipy.integrate import solve_bvp, cumulative_trapezoid
    from scipy.optimize import root, brentq
    import torch
    from torch import nn
    from torch.nn import functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import TwoSlopeNorm
    from tqdm import tqdm

    # Resolve repositories from this script, independent of the shell working directory.
    KNEEP_ROOT = Path(__file__).resolve().parents[1]
    LRAMB_ROOT = (args.lramb_root.expanduser().resolve() if args.lramb_root else KNEEP_ROOT.parent/"LRAMB")
    sys.path.insert(0, str(LRAMB_ROOT))
    from lramb.model import Parameters, KernelSpec, ActiveModelB

    torch.set_num_threads(args.cpu_threads)
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                         "animation.embed_limit": 100})
    print("KNEEP:", KNEEP_ROOT)
    print("LRAMB:", LRAMB_ROOT)

    # ---- Editable experiment settings ----
    SEED = 3                         # 단 하나의 model/master seed
    DEVICE = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if DEVICE.type == "cuda":
        torch.cuda.set_device(DEVICE)

    GRID = dict(n=32, dx=1.0)
    BASE = dict(a=-1.0, b=1.0, kappa=1.0, mobility=1.0,
                g=0.3, ell=1.0, lam=0.8, D=0.1, radius=8.0)
    PHYS = dict(**GRID, **{k:v for k,v in BASE.items() if k != "radius"},
                dt=0.01, kernel="poisson", alpha=1.0)
    # mean is set from the solved circular initial field below, not forced to zero.
    INITIALIZATION = "periodic_stationary"  # "local_bvp": quench from the local solution
    BVP = dict(nodes=400, tol=1e-5, max_nodes=12000)
    ROOT = dict(tol=1e-7, maxiter=100, continuation_steps=8)
    DATA = dict(train_members=128, validation_members=16, test_members=1,
                train_steps=1024, validation_steps=1024, test_steps=5000,
                burn_steps=100000, ensemble_batch=128, trace_stride=100)
    MODEL = dict(n_channel=128, n_hidden=2, max_distance=3,
                 shell_force_bias=False, activation="elu")
    TRAIN = dict(iterations=5000, batch_size=2048, microbatch=128,
                 evaluation_batch=256, lr=1e-3, weight_decay=0.0,
                 alpha=-0.5, clip_norm=1.0, record_every=100,
                 scheduler="none", plateau_factor=0.5, plateau_patience=3,
                 plateau_threshold=1e-3, min_lr=1e-5)
    LIVE_SMOOTHING = 0.5               # display only; checkpoint uses raw full validation
    RUN_TRAINING = args.stage == "all"               # False: existing matching best checkpoint만 평가
    OUTPUT_ROOT = (args.output_dir.expanduser().resolve() if args.output_dir
                   else KNEEP_ROOT / "results" / "lramb_single_run")
    FIT_OUTPUT_ROOT = args.fit_output_dir.expanduser().resolve() if args.fit_output_dir else None           # e.g. Path('/data/user1/lramb_fits'); data cache stays at OUTPUT_ROOT

    # JSON changes only explicitly supplied keys; unspecified values keep notebook defaults.
    if args.config:
        overrides = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
        groups = dict(GRID=GRID, BASE=BASE, DATA=DATA, MODEL=MODEL, TRAIN=TRAIN, BVP=BVP, ROOT=ROOT)
        allowed = set(groups) | {"SEED", "INITIALIZATION", "LIVE_SMOOTHING", "dt"}
        if not isinstance(overrides, dict) or set(overrides)-allowed:
            raise ValueError(f"Config must contain only: {sorted(allowed)}")
        for name, target in groups.items():
            if name in overrides:
                values = overrides[name]
                if not isinstance(values, dict) or set(values)-set(target):
                    raise ValueError(f"{name} override must use keys {sorted(target)}")
                target.update(values)
        SEED = overrides.get("SEED", SEED)
        INITIALIZATION = overrides.get("INITIALIZATION", INITIALIZATION)
        LIVE_SMOOTHING = overrides.get("LIVE_SMOOTHING", LIVE_SMOOTHING)
        PHYS = dict(**GRID, **{k:v for k,v in BASE.items() if k != "radius"},
                    dt=overrides.get("dt", PHYS["dt"]), kernel="poisson", alpha=1.0)

    if PHYS["n"] < 4 or PHYS["n"] % 2 or PHYS["a"] >= 0 or PHYS["b"] <= 0:
        raise ValueError("Use even n >= 4 and a<0, b>0 for a circular double-well profile.")
    if PHYS["kappa"] <= 0:
        raise ValueError("The BVP requires kappa>0.")
    if INITIALIZATION not in ("periodic_stationary", "local_bvp"):
        raise ValueError("Choose periodic_stationary or local_bvp.")
    if not 0 <= MODEL["max_distance"] < PHYS["n"] // 2:
        raise ValueError("Keep max_distance below n/2 to avoid periodic duplicate offsets.")
    if min(DATA.values()) < 1 or min(TRAIN["microbatch"], TRAIN["batch_size"],
                                   TRAIN["iterations"], TRAIN["record_every"],
                                   TRAIN["evaluation_batch"]) < 1:
        raise ValueError("Counts and intervals must be positive.")
    if TRAIN["scheduler"] not in ("none", "plateau"):
        raise ValueError("scheduler must be none or plateau")
    if not -1 < TRAIN["alpha"] <= 0:
        raise ValueError("This demo uses -1 < alpha <= 0.")
    if not 0 <= LIVE_SMOOTHING < 1:
        raise ValueError("LIVE_SMOOTHING must lie in [0,1).")


    n, dt, dV = PHYS["n"], PHYS["dt"], PHYS["dx"] ** 2

    # ---- Circular boundary-value initialization ----
    # Same radial BVP and periodic continuation equations as poisson_sanity.
    # Residuals and solver status are diagnostics, not strict acceptance assertions.
    def initialize_circle():
        n, dx = GRID["n"], GRID["dx"]
        length, rb = n*dx, n*dx/2
        a, b, kappa, radius = (BASE[k] for k in ("a","b","kappa","radius"))
        phi0, width = math.sqrt(-a/b), math.sqrt(2*kappa/(-a))
        mean_disk = phi0*(2*(radius/rb)**2-1)
        target_integral = mean_disk*rb**2/2
        r = np.linspace(0,rb,BVP["nodes"])
        singular = np.zeros((3,3))
        singular[1,1] = -1.

        def equation(r, y, p):
            phi, derivative, integral = y
            return np.vstack((derivative,(a*phi+b*phi**3-p[0])/kappa,r*phi))

        def boundary(left, right, p):
            return np.array([left[1],right[1],left[2],
                             (right[2]-target_integral)/rb**2])

        candidates = []
        for factor in (1., .9, 1.1):
            z = (factor*radius-r)/width
            guess_phi = phi0*np.tanh(z)
            guess = np.vstack((guess_phi,-phi0/width*(1-np.tanh(z)**2),
                               cumulative_trapezoid(r*guess_phi,r,initial=0)))
            solution = solve_bvp(equation,boundary,r,guess,
                                p=[kappa*phi0/(radius*width)],S=singular,
                                tol=BVP["tol"],bc_tol=BVP["tol"],max_nodes=BVP["max_nodes"])
            values = solution.sol(np.linspace(0,rb,2001))[0]
            residual = float(np.max(solution.rms_residuals))
            # Keep a finite positive-core/negative-exterior branch, without tight
            # residual/contrast/monotonicity cutoffs.
            if np.isfinite(values).all() and values[0]>0 and values[-1]<0:
                candidates.append((solution,residual))
                if solution.success:
                    break
        if not candidates:
            raise RuntimeError("BVP returned no finite circular-droplet branch; inspect BVP settings.")
        solution, residual = min(candidates,key=lambda item: (not item[0].success,item[1]))
        radial_info = dict(success=bool(solution.success), message=str(solution.message),
                           max_rms_residual=residual,
                           boundary_error=float(np.max(abs(boundary(solution.y[:,0],
                                                                      solution.y[:,-1],solution.p)))),
                           mu=float(solution.p[0]), nominal_radius=radius,
                           crossing_radius=float(brentq(lambda x:float(solution.sol(x)[0]),0,rb)))
        print("Radial BVP:", radial_info)
        if not solution.success:
            print("BVP status is nonconverged; using the finite droplet candidate shown above.")

        coord = np.arange(n)*dx-length/2
        x,y = np.meshgrid(coord,coord)
        distance = np.hypot(x,y)
        local_field = solution.sol(np.minimum(distance,rb).ravel())[0].reshape(n,n)
        field = local_field.copy()
        target_mean = float(field.mean())
        wave = 2*np.pi*np.fft.fftfreq(n,d=dx)
        qx,qy = np.meshgrid(2*np.sin(wave*dx/2)/dx,2*np.sin(wave*dx/2)/dx)
        q2_init = qx*qx+qy*qy
        multiplier = KernelSpec("poisson",ell=BASE["ell"]).multiplier(np.sqrt(q2_init))

        def chemical_potential(phi, g, lam):
            lap = sum(np.roll(phi,1,ax)+np.roll(phi,-1,ax)-2*phi
                      for ax in (-2,-1))/dx**2
            grad2 = sum((np.roll(phi,1,ax)-phi)**2+(np.roll(phi,-1,ax)-phi)**2
                        for ax in (-2,-1))/(2*dx**2)
            convolution = np.fft.ifft2(multiplier*np.fft.fft2(phi)).real
            return a*phi+b*phi**3-kappa*lap+g*convolution+lam*grad2

        continuation = []
        if INITIALIZATION == "periodic_stationary":
            previous_mu = float(chemical_potential(field,0.,0.).mean())
            for fraction in np.linspace(0.,1.,ROOT["continuation_steps"]+1):
                g,lam = fraction*BASE["g"],fraction*BASE["lam"]
                def periodic_equation(z):
                    phi = z[:-1].reshape(n,n)
                    return np.r_[(chemical_potential(phi,g,lam)-z[-1]).ravel(),
                                 n*n*(phi.mean()-target_mean)]
                solved = root(periodic_equation,np.r_[field.ravel(),previous_mu],method="krylov",
                              options=dict(fatol=ROOT["tol"],maxiter=ROOT["maxiter"]))
                if not np.isfinite(solved.x).all():
                    raise FloatingPointError("Nonfinite periodic initializer; inspect ROOT settings.")
                field = solved.x[:-1].reshape(n,n)
                previous_mu = float(solved.x[-1])
                info = dict(fraction=float(fraction),success=bool(solved.success),
                            residual=float(np.max(abs(periodic_equation(solved.x)))),
                            mass_offset=float(field.mean()-target_mean))
                continuation.append(info)
                print("Periodic continuation:", info)
                if not solved.success:
                    print("  Solver status:", solved.message, "| continuing with this finite iterate.")
        mu = chemical_potential(field,BASE["g"],BASE["lam"])
        initial_info = dict(radial=radial_info,continuation=continuation,
                            initialization=INITIALIZATION,target_mean=target_mean,
                            actual_mean=float(field.mean()),mass_offset=float(field.mean()-target_mean),
                            mu_spread=float(np.max(abs(mu-mu.mean()))),
                            centre=float(field[n//2,n//2]),exterior=float(field[0,0]))
        print("Initial field diagnostics:", initial_info | {"continuation":"see above","radial":"see above"})
        if field[n//2,n//2]<=0 or field[0,0]>=0:
            print("Initializer no longer has a positive core and negative exterior; inspect the preview.")
        return np.ascontiguousarray(field), local_field, initial_info

    INITIAL_FIELD, LOCAL_BVP_FIELD, INITIAL_INFO = initialize_circle()
    # Do not shift the BVP plateaus to impose mean=0.
    PHYS["mean"] = float(INITIAL_FIELD.mean())
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    bound = max(float(abs(LOCAL_BVP_FIELD).max()),float(abs(INITIAL_FIELD).max()))
    for ax,field,title in zip(axes,(LOCAL_BVP_FIELD,INITIAL_FIELD),
                             ("Radial BVP embedding","Chosen periodic initialization")):
        im = ax.imshow(field,origin="lower",extent=(0,n*GRID["dx"],0,n*GRID["dx"]),
                       cmap="RdBu_r",vmin=-bound,vmax=bound)
        ax.set(title=title,xlabel="x",ylabel="y")
        ax.grid(False)
    fig.colorbar(im,ax=axes,label="phi")
    initialization_figure = fig
    plt.close(fig)

    # ---- Cache identity and output paths ----
    def signature(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    DATA_META = dict(format="single-demo-circle-bvp-batched-em-midpoint-v2", physics=PHYS,
                     data=DATA, master_seed=SEED, device_type=DEVICE.type,
                     initialization=dict(kind=INITIALIZATION, radius=BASE["radius"], bvp=BVP, root=ROOT,
                         field_sha256=hashlib.sha256(INITIAL_FIELD.tobytes()).hexdigest()),
                     torch_version=str(torch.__version__),
                     operator_source_sha256=hashlib.sha256(
                         (LRAMB_ROOT / "lramb" / "model.py").read_bytes()).hexdigest())
    DATA_ID = signature(DATA_META)
    DATA_DIR = OUTPUT_ROOT / ("data_" + DATA_ID[:12])
    FIT_META = dict(format="single-demo-shellforce-v1", data_signature=DATA_ID,
                   seed=SEED, model=MODEL, training=TRAIN)
    FIT_ID = signature(FIT_META)
    FIT_DIR = (DATA_DIR if FIT_OUTPUT_ROOT is None else Path(FIT_OUTPUT_ROOT)) / ("fit_" + FIT_ID[:12])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PATH, LAST_PATH = FIT_DIR / "best.pt", FIT_DIR / "last.pt"

    n, dt, dV = PHYS["n"], PHYS["dt"], PHYS["dx"] ** 2
    field_bytes = sum(DATA[f"{s}_members"] * (DATA[f"{s}_steps"] + 1) * n*n*4
                      for s in ("train", "validation", "test"))
    medium_bytes = DATA["test_members"] * DATA["test_steps"] * (n*n + 3) * 8
    print("Device:", DEVICE, "| sole condition:", {k: PHYS[k] for k in ("kernel", "g", "ell")})
    print("Field + reference arrays:", round((field_bytes + medium_bytes) / 1e9, 3), "GB")
    print("Data:", DATA_DIR)
    print("Fit:", FIT_DIR)
    print("Train pairs:", DATA["train_members"] * DATA["train_steps"])
    print("Burn-in:", DATA["burn_steps"], "steps =", DATA["burn_steps"]*dt, "physical time")
    np.savez(DATA_DIR / "initial_state.npz", field=INITIAL_FIELD,
             local_bvp_field=LOCAL_BVP_FIELD, metadata=json.dumps(INITIAL_INFO))

    initialization_figure.savefig(DATA_DIR/"initial_profile.png", dpi=150)
    resolved_config = dict(GRID=GRID, BASE=BASE, DATA=DATA, MODEL=MODEL, TRAIN=TRAIN,
                           BVP=BVP, ROOT=ROOT, SEED=SEED, INITIALIZATION=INITIALIZATION,
                           LIVE_SMOOTHING=LIVE_SMOOTHING, dt=dt)
    (FIT_DIR/"resolved_config.json").write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
    (FIT_DIR/"run_settings.json").write_text(json.dumps(
        dict(data=DATA_META, fit=FIT_META, device=str(DEVICE), stage=args.stage), indent=2), encoding="utf-8")

    # ---- Euler-Maruyama and midpoint medium entropy ----
    parameters = Parameters(**{k: PHYS[k] for k in
        ("n", "dx", "dt", "a", "b", "kappa", "lam", "mobility", "D", "mean")},
        seed=SEED)
    operator = ActiveModelB(parameters, KernelSpec(PHYS["kernel"],
        strength=PHYS["g"], ell=PHYS["ell"], alpha=PHYS["alpha"]))
    kw = dict(device=DEVICE, dtype=torch.float64)
    Q2 = torch.as_tensor(operator.q2[:, :n//2+1], **kw)
    K = torch.as_tensor(operator.K[:, :n//2+1], **kw)
    LINEAR = torch.as_tensor(operator.linear_mu[:, :n//2+1], **kw)
    NOISE = torch.as_tensor(operator.noise_scale[:, :n//2+1], **kw)
    MULT = torch.full((n//2+1,), 2.0, **kw)
    MULT[0] = MULT[-1] = 1.0

    # Solve once, broadcast the same reference circle to every ensemble member.
    INITIAL_TENSOR = torch.as_tensor(INITIAL_FIELD, **kw)

    def initial_batch(count):
        return INITIAL_TENSOR[None].expand(count, n, n).clone()

    def neighbours(phi):
        left, right = torch.roll(phi, 1, -2), torch.roll(phi, -1, -2)
        down, up = torch.roll(phi, 1, -1), torch.roll(phi, -1, -1)
        lap = (left + right + down + up - 4*phi) / dV
        grad2 = ((left-phi).square() + (right-phi).square()
                 + (down-phi).square() + (up-phi).square()) / (2*dV)
        return lap, grad2

    def advance(phi, ph, rng):
        _, grad2 = neighbours(phi)
        nonlinear = PHYS["b"] * phi.pow(3) + PHYS["lam"] * grad2
        normal = torch.randn(phi.shape, generator=rng, **kw)
        new_hat = ph - dt * PHYS["mobility"] * Q2 * (
            LINEAR * ph + torch.fft.rfft2(nonlinear, norm="ortho"))
        new_hat = new_hat + NOISE * torch.fft.rfft2(normal, norm="ortho")
        # Enforce real-field symmetry at the two rFFT boundary columns.
        for col in (0, -1):
            value = new_hat[..., col]
            new_hat[..., col] = (value + torch.roll(value.flip(-1), 1, -1).conj()) / 2
        new_hat[..., 0, 0] = ph[..., 0, 0]
        new = torch.fft.irfft2(new_hat, s=(n, n), norm="ortho")
        canonical = torch.fft.rfft2(new, norm="ortho")
        return new, canonical

    def midpoint_entropy(old, new, old_hat, new_hat):
        mid = (old + new) / 2
        lap, grad2 = neighbours(mid)
        local = PHYS["a"]*mid + PHYS["b"]*mid.pow(3) - PHYS["kappa"]*lap
        nonlocal_mu = PHYS["g"] * torch.fft.irfft2(
            K * (old_hat + new_hat) / 2, s=(n, n), norm="ortho")
        mu = torch.stack((local, nonlocal_mu, PHYS["lam"]*grad2), dim=1)
        # Integrated entropy density, NOT a rate: [B, 3, H, W].
        return -mu * (new-old)[:, None] / PHYS["D"]

    def diagnostics(phi, ph):
        power = ph.abs().square() * MULT
        bulk = dV * (PHYS["a"]/2*phi.square() + PHYS["b"]/4*phi.pow(4)).sum((-2,-1))
        gradient = dV * PHYS["kappa"]/2 * (Q2*power).sum((-2,-1))
        long_range = dV * PHYS["g"]/2 * (K*power).sum((-2,-1))
        lowq = ph[..., 0, 1].abs().square() + ph[..., 1, 0].abs().square()
        return torch.stack((phi.mean((-2,-1)), phi.var((-2,-1), correction=0),
                            bulk+gradient+long_range, lowq), dim=-1)

    # ---- Generate or reuse the ensemble data ----
    @torch.inference_mode()
    def generate_split(split):
        members, steps = DATA[f"{split}_members"], DATA[f"{split}_steps"]
        burn, chunk = DATA["burn_steps"], 64
        # Fixed, disjoint streams derived from ONE master seed.
        split_seed = SEED + {"train": 10000, "validation": 20000, "test": 30000}[split]
        rng = torch.Generator(device=DEVICE).manual_seed(split_seed)
        video = np.lib.format.open_memmap(DATA_DIR / f"{split}_video.npy", mode="w+",
            dtype=np.float32, shape=(members, steps+1, n, n))
        if split == "test":
            ref = np.lib.format.open_memmap(DATA_DIR / "test_medium_increment.npy",
                mode="w+", dtype=np.float64, shape=(members, steps, 3))
            maps = np.lib.format.open_memmap(DATA_DIR / "test_medium_density.npy",
                mode="w+", dtype=np.float64, shape=(members, steps, n, n))

        for first in range(0, members, DATA["ensemble_batch"]):
            last = min(first + DATA["ensemble_batch"], members)
            size = last-first
            phi = initial_batch(size)
            ph = torch.fft.rfft2(phi, norm="ortho")
            target_mass = phi.mean((-2,-1))
            max_mass_drift = torch.zeros((), **kw)
            field_buffer = torch.empty((size, min(chunk, steps+1), n, n),
                                       device=DEVICE, dtype=torch.float32)
            if split == "test":
                map_buffer = torch.empty((size, min(chunk, steps), n, n), **kw)
                increment = torch.empty((size, steps, 3), **kw)
            trace_times, trace_fields, trace_diag = [], [], []

            def record_trace(step):
                if split == "test" and first == 0:
                    trace_times.append((step-burn)*dt)
                    trace_fields.append(phi[0].float().clone())
                    trace_diag.append(diagnostics(phi[:1], ph[:1])[0].clone())

            def record_frame(index):
                capacity = field_buffer.shape[1]
                slot = index % capacity
                field_buffer[:, slot].copy_(phi)
                if slot == capacity-1 or index == steps:
                    video[first:last, index-slot:index+1] = field_buffer[:, :slot+1].cpu().numpy()

            record_trace(0)
            iterator = tqdm(range(1, burn+steps+1),
                desc=f"{split} members {first}:{last}", leave=False, mininterval=1)
            for step in iterator:
                new, new_hat = advance(phi, ph, rng)
                if split == "test" and step > burn:
                    j = step-burn-1
                    ds = midpoint_entropy(phi, new, ph, new_hat)
                    increment[:, j] = ds.sum((-2,-1)) * dV
                    slot = j % map_buffer.shape[1]
                    map_buffer[:, slot] = ds.sum(1)
                    if slot == map_buffer.shape[1]-1 or j == steps-1:
                        maps[first:last, j-slot:j+1] = map_buffer[:, :slot+1].cpu().numpy()
                phi, ph = new, new_hat
                if step % 100 == 0 or step == burn+steps:
                    if not bool(torch.isfinite(phi).all()):
                        raise FloatingPointError(f"{split}: unstable field at step {step}; reduce dt.")
                    max_mass_drift = torch.maximum(max_mass_drift,
                                                    (phi.mean((-2,-1))-target_mass).abs().max())
                if step % DATA["trace_stride"] == 0 or step in (burn, burn+steps):
                    record_trace(step)
                if step >= burn:
                    record_frame(step-burn)
            print(f"{split} members {first}:{last}: max sampled mean drift="
                  f"{float(max_mass_drift.cpu()):.3e}")
            if split == "test":
                ref[first:last] = increment.cpu().numpy()
                if first == 0:
                    np.savez(DATA_DIR / "test_trace.npz", time=np.array(trace_times),
                             fields=torch.stack(trace_fields).cpu().numpy(),
                             diagnostics=torch.stack(trace_diag).cpu().numpy())
        video.flush()
        if split == "test":
            ref.flush()
            maps.flush()

    required = ["train_video.npy", "validation_video.npy", "test_video.npy",
                "test_medium_increment.npy", "test_medium_density.npy", "test_trace.npz"]
    manifest_path = DATA_DIR / "data.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    cache_ready = (manifest.get("signature") == DATA_ID and manifest.get("complete", False)
                   and all((DATA_DIR / name).is_file() for name in required))
    if not cache_ready and args.stage == "evaluate":
        raise FileNotFoundError(f"Evaluation requires a complete matching data cache: {DATA_DIR}")
    if not cache_ready:
        manifest = dict(signature=DATA_ID, config=DATA_META, complete=False)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        started = time.perf_counter()
        for split in ("train", "validation", "test"):
            generate_split(split)
        manifest["complete"] = True
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
        print(f"Generated one condition in {time.perf_counter()-started:.1f} s")
    else:
        print("Using the matching single-condition data cache.")

    train_video = np.load(DATA_DIR / "train_video.npy", mmap_mode="r")
    validation_video = np.load(DATA_DIR / "validation_video.npy", mmap_mode="r")
    test_video = np.load(DATA_DIR / "test_video.npy", mmap_mode="r")
    medium_increment = np.load(DATA_DIR / "test_medium_increment.npy", mmap_mode="r")
    medium_density_increment = np.load(DATA_DIR / "test_medium_density.npy", mmap_mode="r")
    with np.load(DATA_DIR / "test_trace.npz") as stored:
        trace = {name: stored[name].copy() for name in stored.files}
    print("Train / validation / test shapes:",
          train_video.shape, validation_video.shape, test_video.shape)

    # ---- Profile movie ----
    MOVIE_START = trace["time"][0]       # 0.0이면 production부터
    MOVIE_STOP = trace["time"][-1]
    MOVIE_MAX_FRAMES = args.movie_max_frames
    MOVIE_FPS = args.movie_fps
    selected = np.flatnonzero((trace["time"] >= MOVIE_START) &
                             (trace["time"] <= MOVIE_STOP))
    if selected.size == 0 or MOVIE_MAX_FRAMES < 1 or MOVIE_FPS <= 0:
        raise ValueError("Choose a nonempty movie interval and positive frame/FPS values.")
    if selected.size > MOVIE_MAX_FRAMES:
        selected = selected[np.unique(np.linspace(0, selected.size-1,
                                                  MOVIE_MAX_FRAMES).astype(int))]
    frames, times = trace["fields"][selected], trace["time"][selected]
    extent = (0, n*PHYS["dx"], 0, n*PHYS["dx"])
    vmax = max(float(np.abs(frames).max()), 1e-12)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(frames[0], origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="phi")
    ax.set(xlabel="x", ylabel="y")
    ax.grid(False)
    title = ax.set_title("")

    def update_profile(i):
        im.set_data(frames[i])
        stage = "burn-in" if times[i] < 0 else "production"
        title.set_text(f"{stage}: t={times[i]:.2f} | g={PHYS['g']}, ell={PHYS['ell']}")
        return im, title

    if args.movie != "none":
        profile_animation = animation.FuncAnimation(
            fig, update_profile, frames=len(frames), interval=1000/MOVIE_FPS, blit=False)
        movie_format = args.movie
        if movie_format == "auto":
            movie_format = "mp4" if animation.writers.is_available("ffmpeg") else "html"
        if movie_format == "mp4":
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError("MP4 needs ffmpeg; use --movie html or --movie auto.")
            movie_path = FIT_DIR/"profile.mp4"
            profile_animation.save(movie_path, fps=MOVIE_FPS, dpi=120)
        else:
            movie_path = FIT_DIR/"profile.html"
            movie_path.write_text("<!doctype html><meta charset='utf-8'>"+
                                  profile_animation.to_jshtml(fps=MOVIE_FPS), encoding="utf-8")
        print("Profile movie:", movie_path, flush=True)
    plt.close(fig)

    # ---- Burn-in and steady diagnostics ----
    # Steady diagnostics: four independent axes, same physical time convention.
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    labels = ("spatial mean phi", "spatial variance", "conservative free energy",
              "power in first x/y Fourier modes")
    for col, ax in enumerate(axes.flat):
        ax.plot(trace["time"], trace["diagnostics"][:, col], lw=1)
        ax.axvline(0, color="black", linestyle="--", alpha=0.6)
        ax.set(ylabel=labels[col], xlabel="time relative to end of burn-in")
    fig.suptitle("First held-out trajectory: inspect late burn-in and production drift")
    fig.tight_layout()
    fig.savefig(FIT_DIR / "steady_diagnostics.png", dpi=150)
    plt.close(fig)
    print("No automatic steady-state pass/fail is inferred from these plots.")

    if args.stage == "generate":
        print("Data generation and profile diagnostics complete:", DATA_DIR, flush=True)
        print("Figures/movie:", FIT_DIR, flush=True)
        return

    # ---- ShellForce model ----
    class ShellAbsoluteConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, radius, bias=False):
            super().__init__()
            self.radius = radius
            side = 2*radius+1
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels, side, side))
            self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
            mask = torch.zeros(1, 1, side, side)
            mask[..., 0, :] = mask[..., -1, :] = 1
            mask[..., :, 0] = mask[..., :, -1] = 1
            self.register_buffer("mask", mask)
            nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        def forward(self, midpoint):
            r = self.radius
            return F.conv2d(F.pad(midpoint, (r,r,r,r), mode="circular"),
                            self.weight*self.mask, bias=self.bias)

    class ShellBranch(nn.Module):
        def __init__(self, radius, width, depth, shell_force_bias, activation):
            super().__init__()
            make_activation = {"elu": lambda: nn.ELU(inplace=True),
                               "relu": lambda: nn.ReLU(inplace=True),
                               "tanh": nn.Tanh}[activation]
            bias = radius == 0 or shell_force_bias
            self.input = (nn.Conv2d(1, width, 1, bias=True) if radius == 0 else
                          ShellAbsoluteConv2d(1, width, radius, bias=shell_force_bias))
            self.activation = make_activation()
            layers = []
            for _ in range(max(depth-1, 0)):
                layers += [nn.Conv2d(width, width, 1, bias=bias), make_activation()]
            layers += [nn.Conv2d(width, 1, 1, bias=bias)]
            self.head = nn.Sequential(*layers)

        def forward(self, midpoint):
            return self.head(self.activation(self.input(midpoint)))

    class DemoShellForce(nn.Module):
        def __init__(self, n_channel, n_hidden, max_distance, shell_force_bias, activation):
            super().__init__()
            self.branches = nn.ModuleList([
                ShellBranch(r, n_channel, n_hidden, shell_force_bias, activation)
                for r in range(max_distance+1)])
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, pair, return_maps=False):
            # [B, 2, H, W], same scalar-field convention as Corr_AMB2D_ShellForce.
            midpoint = 0.5*(pair[:, :1] + pair[:, 1:2])
            delta = pair[:, 1:2] - pair[:, :1]
            maps = [branch(midpoint)*delta for branch in self.branches]
            if return_maps:
                return torch.cat(maps, dim=1)
            return torch.cat([value.mean((-2,-1)) for value in maps], dim=1)

    # ---- Train normalization and objective ----
    def fetch_pairs(video, member, time_index):
        # Advanced indexing makes compact copies from disk-backed arrays.
        pair = np.stack((video[member, time_index], video[member, time_index+1]), axis=1)
        return torch.from_numpy(pair).to(device=DEVICE, dtype=torch.float32)

    def train_moments(video):
        count, mean, m2 = 0, 0.0, 0.0
        for member in range(video.shape[0]):
            for start in range(0, video.shape[1], 128):
                block = np.asarray(video[member, start:start+128], dtype=np.float64)
                k, avg = block.size, float(block.mean())
                delta = avg-mean
                total = count+k
                m2 += float(((block-avg)**2).sum()) + delta*delta*count*k/total
                mean += delta*k/total
                count = total
        return mean, math.sqrt(max(m2/(count-1), 1e-24))

    normal_mean, normal_std = train_moments(train_video)
    def normalize(pair):
        return (pair-normal_mean) / normal_std

    def alpha_terms(score):
        alpha = TRAIN["alpha"]
        if alpha == 0:
            return -score + torch.expm1(-score)
        return -torch.expm1(alpha*score)/alpha + torch.expm1(-(1+alpha)*score)/(1+alpha)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = DemoShellForce(**MODEL).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"],
                                 weight_decay=TRAIN["weight_decay"])
    scheduler = None
    if TRAIN["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=TRAIN["plateau_factor"],
            patience=TRAIN["plateau_patience"], threshold_mode="abs",
            threshold=TRAIN["plateau_threshold"], min_lr=TRAIN["min_lr"])
    sampler = torch.Generator(device="cpu").manual_seed(SEED)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Train mean/std:", normal_mean, normal_std)

    @torch.inference_mode()
    def validation_loss():
        model.eval()
        steps = validation_video.shape[1]-1
        total = validation_video.shape[0]*steps
        loss_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
        for start in range(0, total, TRAIN["evaluation_batch"]):
            flat = np.arange(start, min(start+TRAIN["evaluation_batch"], total))
            pair = normalize(fetch_pairs(validation_video, flat//steps, flat%steps))
            loss_sum += alpha_terms(model(pair).sum(1)).double().sum()
        return float(loss_sum.cpu()) / total

    # ---- Training and checkpoint resume ----
    history = dict(iteration=[], train=[], validation=[], lr=[])
    best_loss, best_iteration = float("inf"), 0
    checkpoint_recovery = None

    def cpu_snapshot(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu_snapshot(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu_snapshot(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu_snapshot(v) for v in value)
        return value

    def save_checkpoint(path, iteration, val_loss):
        nonlocal checkpoint_recovery
        path = Path(path)
        payload = cpu_snapshot(dict(signature=FIT_ID, settings=FIT_META,
            state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            iteration=iteration,
            validation_loss=val_loss, mean=normal_mean, std=normal_std,
            history={k: list(v) for k,v in history.items()},
            sampler_state=sampler.get_state(),
            scheduler_state=None if scheduler is None else scheduler.state_dict()))
        checkpoint_recovery = payload   # independent CPU copy survives write errors
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
            temporary = Path(name)
            with os.fdopen(fd, 'wb') as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except (OSError, RuntimeError) as error:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                free = f'{shutil.disk_usage(path.parent).free / 1e9:.3f} GB filesystem free'
            except OSError:
                free = 'free space unavailable'
            raise OSError(
                f"Checkpoint write failed: {path} ({free}). Check disk space, quota, permissions "
                "and filesystem I/O. Previous completed checkpoints are preserved. "
                "Restart with --resume after resolving the storage problem."
            ) from error

    def ema(values, smoothing=LIVE_SMOOTHING):
        result = []
        for value in values:
            result.append(value if not result else
                          smoothing*result[-1] + (1-smoothing)*value)
        return result

    start_iteration = 1
    if args.resume:
        if not LAST_PATH.exists():
            raise FileNotFoundError(f"No matching last checkpoint to resume: {LAST_PATH}")
        restored = torch.load(LAST_PATH, map_location="cpu", weights_only=True)
        if restored["signature"] != FIT_ID:
            raise ValueError("Resume checkpoint settings/data do not match.")
        model.load_state_dict(restored["state_dict"])
        optimizer.load_state_dict(restored["optimizer"])
        normal_mean, normal_std = float(restored["mean"]), float(restored["std"])
        sampler.set_state(restored["sampler_state"].cpu())
        history = {key:list(values) for key,values in restored["history"].items()}
        if scheduler is not None and restored.get("scheduler_state") is not None:
            scheduler.load_state_dict(restored["scheduler_state"])
        start_iteration = int(restored["iteration"])+1
        if BEST_PATH.exists():
            best = torch.load(BEST_PATH, map_location="cpu", weights_only=True)
            if best["signature"] != FIT_ID:
                raise ValueError("Best checkpoint does not match the resumed run.")
            best_loss, best_iteration = float(best["validation_loss"]), int(best["iteration"])
            del best
        else:
            best_loss, best_iteration = float(restored["validation_loss"]), int(restored["iteration"])
            # Keep evaluation usable even when only last.pt was copied.
            save_checkpoint(BEST_PATH, best_iteration, best_loss)
        del restored
        print(f"Resuming at iteration {start_iteration}; target={TRAIN['iterations']}", flush=True)

    if RUN_TRAINING and not args.resume and (LAST_PATH.exists() or BEST_PATH.exists()):
        raise FileExistsError(f"Matching fit already exists: {FIT_DIR}. Use --resume to continue "
                              "or --stage evaluate for plotting/inference.")
    if RUN_TRAINING:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        train_raw, = axes[0].plot([], [], color="C0", alpha=0.2, label="sampled train (raw)")
        val_raw, = axes[0].plot([], [], color="C1", alpha=0.25, label="full validation (raw)")
        train_line, = axes[0].plot([], [], color="C0", label="train EMA (display)")
        val_line, = axes[0].plot([], [], color="C1", label="validation EMA (display)")
        lr_line, = axes[1].plot([], [], color="C2")
        axes[0].set(xlabel="optimizer iteration", ylabel="alpha-divergence objective")
        axes[0].legend(fontsize=8)
        axes[1].set(xlabel="optimizer iteration", ylabel="learning rate", yscale="log")
        fig.tight_layout()
        plt.close(fig)
        # Populate existing history even if a resumed fit has already finished.
        train_raw.set_data(history["iteration"], history["train"])
        val_raw.set_data(history["iteration"], history["validation"])
        train_line.set_data(history["iteration"], ema(history["train"]))
        val_line.set_data(history["iteration"], ema(history["validation"]))
        lr_line.set_data(history["iteration"], history["lr"])
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        for iteration in tqdm(range(start_iteration, TRAIN["iterations"]+1), desc="single-seed fit"):
            model.train()
            member = torch.randint(train_video.shape[0], (TRAIN["batch_size"],),
                                   generator=sampler).numpy()
            time_index = torch.randint(train_video.shape[1]-1, (TRAIN["batch_size"],),
                                       generator=sampler).numpy()
            optimizer.zero_grad(set_to_none=True)
            train_loss = torch.zeros((), device=DEVICE)
            lr_used = optimizer.param_groups[0]["lr"]
            for start in range(0, TRAIN["batch_size"], TRAIN["microbatch"]):
                end = min(start+TRAIN["microbatch"], TRAIN["batch_size"])
                pair = normalize(fetch_pairs(train_video, member[start:end], time_index[start:end]))
                loss = alpha_terms(model(pair).sum(1)).sum() / TRAIN["batch_size"]
                loss.backward()
                train_loss += loss.detach()
            if not bool(torch.isfinite(train_loss)):
                raise FloatingPointError("Nonfinite objective; inspect field, normalization and LR.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["clip_norm"],
                                           error_if_nonfinite=True)
            optimizer.step()

            if iteration == 1 or iteration % TRAIN["record_every"] == 0 or iteration == TRAIN["iterations"]:
                val_loss = validation_loss()
                if not math.isfinite(val_loss):
                    raise FloatingPointError("Nonfinite validation objective.")
                history["iteration"].append(iteration)
                history["train"].append(float(train_loss.cpu()))
                history["validation"].append(val_loss)
                history["lr"].append(lr_used)
                if scheduler is not None:
                    scheduler.step(val_loss)
                if val_loss < best_loss:
                    best_loss, best_iteration = val_loss, iteration
                    save_checkpoint(BEST_PATH, iteration, val_loss)
                save_checkpoint(LAST_PATH, iteration, val_loss)
                (FIT_DIR / "history.json").write_text(json.dumps(history), encoding="utf-8")

                it = history["iteration"]
                train_raw.set_data(it, history["train"])
                val_raw.set_data(it, history["validation"])
                train_line.set_data(it, ema(history["train"]))
                val_line.set_data(it, ema(history["validation"]))
                lr_line.set_data(it, history["lr"])
                axes[0].set_title(f"seed={SEED}; best validation={best_loss:.5g} at {best_iteration}")
                for ax in axes:
                    ax.relim()
                    ax.autoscale_view()
                fig.savefig(FIT_DIR / "training_loss.png", dpi=150)
                print(f"iteration={iteration} train={history['train'][-1]:.6g} "
                      f"valid={val_loss:.6g} lr={lr_used:.4g} "
                      f"best={best_loss:.6g}@{best_iteration}", flush=True)
        fig.savefig(FIT_DIR / "training_loss.png", dpi=150)
        print("Best iteration:", best_iteration, "| best loss:", best_loss)
    else:
        evaluation_path = BEST_PATH if args.checkpoint == "best" else LAST_PATH
        if not evaluation_path.exists():
            raise FileNotFoundError(f"No matching evaluation checkpoint: {evaluation_path}")
        # Use the full training history even when evaluating the earlier best model.
        history_path = LAST_PATH if LAST_PATH.exists() else evaluation_path
        saved_history = torch.load(history_path, map_location="cpu", weights_only=True)
        if saved_history["signature"] != FIT_ID:
            raise ValueError("Training history checkpoint does not match these settings.")
        history = saved_history["history"]
        del saved_history
        print("Skipped fitting; matching checkpoint will be loaded below.")

    # ---- Raw training history ----
    # Change this cell freely to inspect raw loss, last iterations, or a different smoothing.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["iteration"], history["train"], ".-", alpha=0.6, label="sampled train")
    axes[0].plot(history["iteration"], history["validation"], "-", label="full validation")
    axes[0].set(xlabel="optimizer iteration", ylabel="raw alpha objective",
                title="One fit seed: raw losses")
    axes[0].legend()
    axes[1].plot(history["iteration"], history["lr"])
    axes[1].set(xlabel="optimizer iteration", ylabel="learning rate", yscale="log")
    fig.tight_layout()
    fig.savefig(FIT_DIR/"training_loss_raw.png", dpi=150)
    plt.close(fig)

    # ---- Held-out evaluation ----
    LOAD_BEST = args.checkpoint == "best"
    EVAL_START = 0
    EVAL_STOP = None                 # None: full held-out production interval
    PREDICT_BATCH = TRAIN["evaluation_batch"]

    checkpoint_path = BEST_PATH if LOAD_BEST else LAST_PATH
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if checkpoint["signature"] != FIT_ID:
        raise ValueError("Checkpoint settings/data do not match this notebook configuration.")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normal_mean, normal_std = float(checkpoint["mean"]), float(checkpoint["std"])
    print("Loaded:", checkpoint_path, "| iteration:", checkpoint["iteration"])

    stop = test_video.shape[1]-1 if EVAL_STOP is None else int(EVAL_STOP)
    start = int(EVAL_START)
    if not 0 <= start < stop <= test_video.shape[1]-1 or PREDICT_BATCH < 1:
        raise ValueError("Choose a valid nonempty test interval and positive batch.")
    members, length, shells = test_video.shape[0], stop-start, MODEL["max_distance"]+1
    pred_increment = np.empty((members, length, shells), dtype=np.float64)
    pred_map_sum = torch.zeros((members, shells, n, n), dtype=torch.float64, device=DEVICE)

    with torch.inference_mode():
        for first in tqdm(range(0, members*length, PREDICT_BATCH), desc="held-out inference"):
            flat = np.arange(first, min(first+PREDICT_BATCH, members*length))
            member, relative_time = flat//length, flat%length
            pair = normalize(fetch_pairs(test_video, member, start+relative_time))
            raw = model(pair, return_maps=True)
            pred_increment[member, relative_time] = raw.mean((-2,-1)).double().cpu().numpy()
            # Sum over exactly the same time indices as the increments.
            pred_map_sum.index_add_(0, torch.as_tensor(member, device=DEVICE),
                                    raw.double() / (n*n))
    pred_mean_density = pred_map_sum.cpu().numpy() / (length*dt*dV)
    ref_increment = np.array(medium_increment[:, start:stop], dtype=np.float64)
    medium_mean_density = np.empty((members, n, n), dtype=np.float64)
    for member in range(members):
        medium_mean_density[member] = np.mean(
            medium_density_increment[member, start:stop], axis=0, dtype=np.float64) / dt
    pred_total = pred_increment.sum(-1)
    medium_total = ref_increment.sum(-1)
    duration = length*dt
    radii = np.arange(shells) * PHYS["dx"]

    # Numerical bookkeeping diagnostics; these do not equate the two EP gauges.
    pred_map_rate = pred_mean_density.sum((1,2,3))*dV
    ref_map_rate = medium_mean_density.sum((1,2))*dV
    print("Max pred map-integral residual:",
          np.max(np.abs(pred_map_rate-pred_total.sum(1)/duration)))
    print("Max medium map-integral residual:",
          np.max(np.abs(ref_map_rate-medium_total.sum(1)/duration)))

    # ---- Cumulative and window-averaged EP ----
    WINDOW_STEPS = 64
    PERFORMANCE_TRAJECTORY = 0       # None: test ensemble mean
    if WINDOW_STEPS < 1:
        raise ValueError("WINDOW_STEPS must be positive.")
    if PERFORMANCE_TRAJECTORY is None:
        p = pred_total.mean(0)
        r = medium_total.mean(0)
        curve_label = "test ensemble mean"
    else:
        p = pred_total[PERFORMANCE_TRAJECTORY]
        r = medium_total[PERFORMANCE_TRAJECTORY]
        curve_label = f"test trajectory {PERFORMANCE_TRAJECTORY}"

    edges = np.arange(length+1)*dt
    starts = np.arange(0, length, WINDOW_STEPS)
    ends = np.minimum(starts+WINDOW_STEPS, length)
    window_duration = (ends-starts)*dt
    window_time = (starts+ends)*dt/2
    pred_window = np.add.reduceat(p, starts) / window_duration
    medium_window = np.add.reduceat(r, starts) / window_duration

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(edges, np.r_[0.0, np.cumsum(r)], label="midpoint medium", color="C1")
    axes[0].plot(edges, np.r_[0.0, np.cumsum(p)], label="KNEEP prediction", color="C0")
    axes[0].set(xlabel="elapsed time in selected interval", ylabel="cumulative EP")
    axes[1].plot(window_time, medium_window, label="midpoint medium", color="C1")
    axes[1].plot(window_time, pred_window, label="KNEEP prediction", color="C0")
    axes[1].set(xlabel="elapsed time (window center)", ylabel="window mean EP / time",
                title=f"Windows of up to {WINDOW_STEPS} steps")
    for ax in axes:
        ax.legend()
    fig.suptitle(f"{curve_label}; seed={SEED}, g={PHYS['g']}, ell={PHYS['ell']}")
    fig.tight_layout()
    fig.savefig(FIT_DIR / "performance.png", dpi=150)
    plt.close(fig)

    pred_rate, medium_rate = float(p.sum()/duration), float(r.sum()/duration)
    ratio = pred_rate/medium_rate if abs(medium_rate) > 1e-12 else float("nan")
    print(f"Mean rate: pred={pred_rate:.6g}, medium={medium_rate:.6g}, pred/medium={ratio:.5g}")
    print("These compare different EP gauges; the medium series excludes system entropy.")

    # ---- Time-averaged maps ----
    MAP_TRAJECTORY = 0              # None: ensemble mean
    MAP_CMAP = "RdBu_r"
    MAP_VMAX = None                 # None: common max absolute value
    if MAP_TRAJECTORY is None:
        pred_map = pred_mean_density.sum(1).mean(0)
        medium_map = medium_mean_density.mean(0)
    else:
        pred_map = pred_mean_density[MAP_TRAJECTORY].sum(0)
        medium_map = medium_mean_density[MAP_TRAJECTORY]
    limit = max(float(np.abs(pred_map).max()), float(np.abs(medium_map).max()), 1e-12)
    limit = limit if MAP_VMAX is None else float(MAP_VMAX)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, value, title in zip(axes, (medium_map, pred_map),
                                 ("Midpoint medium", "KNEEP prediction")):
        image = ax.imshow(value, origin="lower", extent=extent, cmap=MAP_CMAP, norm=norm)
        ax.set(xlabel="x", ylabel="y",
               title=f"{title}\nspatial integral={value.sum()*dV:.5g}")
        ax.grid(False)
    fig.colorbar(image, ax=axes, label="time-averaged EP / (time × area)", shrink=0.85)
    fig.savefig(FIT_DIR / "time_averaged_medium_vs_pred.png", dpi=150)
    plt.close(fig)

    # ---- Mean shell spectrum ----
    MEAN_YSCALE = "symlog"          # "linear" or "symlog"
    MEAN_LINTHRESH = 1e-5
    mean_shell_rate = pred_increment.mean((0,1)) / dt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(radii, mean_shell_rate, "o-", color="C0")
    ax.axhline(0, color="black", lw=0.7)
    if MEAN_YSCALE == "symlog":
        ax.set_yscale("symlog", linthresh=MEAN_LINTHRESH)
    elif MEAN_YSCALE != "linear":
        raise ValueError("Use linear or symlog for signed mean EP.")
    ax.set(xlabel="Chebyshev shell radius", ylabel="mean predicted EP / time",
           title="Predicted shell spectrum: time average", xticks=radii)
    fig.tight_layout()
    fig.savefig(FIT_DIR / "pred_spectrum_mean.png", dpi=150)
    plt.close(fig)
    print("Sum of shell means =", mean_shell_rate.sum(),
          "| mean global predicted rate =", pred_total.mean()/dt)

    # ---- Shell fluctuation ----
    FLUCTUATION_WINDOW = 1
    FLUCTUATION_YSCALE = "symlog"    # retains exactly zero variance
    VAR_LINTHRESH = 1e-8
    DIVIDE_BY_RHO_SQUARED = False
    if not isinstance(FLUCTUATION_WINDOW, int) or FLUCTUATION_WINDOW < 1:
        raise ValueError("Choose a positive integer fluctuation window.")
    complete_windows = length // FLUCTUATION_WINDOW
    if complete_windows < 2:
        raise ValueError("Need at least two complete windows for temporal variance.")
    used = complete_windows*FLUCTUATION_WINDOW
    shell_window_rate = pred_increment[:, :used].reshape(
        members, complete_windows, FLUCTUATION_WINDOW, shells).sum(2) / (FLUCTUATION_WINDOW*dt)
    variance_shell = shell_window_rate.var(axis=1, ddof=0).mean(0)
    x, y = radii, variance_shell
    ylabel = "temporal variance of predicted EP rate"
    if DIVIDE_BY_RHO_SQUARED:
        x, y = radii[1:], variance_shell[1:] / radii[1:]**2
        ylabel += " / radius²"
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(x, y, "o-", color="C2")
    if FLUCTUATION_YSCALE == "symlog":
        ax.set_yscale("symlog", linthresh=VAR_LINTHRESH)
    elif FLUCTUATION_YSCALE != "linear":
        raise ValueError("Use linear or symlog to retain zero variance.")
    ax.set(xlabel="Chebyshev shell radius", ylabel=ylabel, xticks=x,
           title=f"Predicted shell fluctuation: window={FLUCTUATION_WINDOW*dt:g} time")
    fig.tight_layout()
    fig.savefig(FIT_DIR / "pred_spectrum_fluctuation.png", dpi=150)
    plt.close(fig)
    print("Local (rho=0) variance:", variance_shell[0])
    print("Total rate variance (includes cross-shell covariance):",
          shell_window_rate.sum(-1).var(axis=1, ddof=0).mean())
    print("Discarded tail for variance:", length-used, "transitions")

    # ---- Saved evaluation arrays ----
    eval_meta = dict(fit_signature=FIT_ID, data_signature=DATA_ID,
                     model_iteration=checkpoint["iteration"], start=start, stop=stop,
                     dt=dt, dx=PHYS["dx"], physics=PHYS,
                     reference="all-component midpoint medium; no system entropy")
    np.savez_compressed(FIT_DIR / "evaluation.npz",
        predicted_increment=pred_increment, medium_increment=ref_increment,
        predicted_mean_density=pred_mean_density, medium_mean_density=medium_mean_density,
        radii=radii, metadata=json.dumps(eval_meta))
    print("Saved notebook outputs:", FIT_DIR)
    # Later, for plotting only:
    # with np.load(FIT_DIR / "evaluation.npz", allow_pickle=False) as data:
    #     pred_increment = data["predicted_increment"].copy()

    print("Single-run experiment complete:", FIT_DIR, flush=True)


if __name__ == "__main__":
    run(parse_args())
