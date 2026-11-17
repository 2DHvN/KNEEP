#!/usr/bin/env python3
"""Standalone Poisson sanity simulation, diagnostics and central x-cut movie.

Run from any directory; no Jupyter/IPython or notebook file is needed.
Examples:
    python scripts/poisson_sanity.py --device cuda:0
    python scripts/poisson_sanity.py --device cpu --config sanity.json --movie none
Config JSON may override GRID, BASE, RUN, BVP, ROOT, RULES, CASES,
INITIALIZATION, REMOVE_ZERO_MODE and MODE_MAX. CASES contains overrides of BASE.
Cached trajectories and numbered PNG figures are written under --output-dir.
"""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--lramb-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--movie", choices=("html", "mp4", "none"), default="html")
    parser.add_argument("--movie-fps", type=int, default=12)
    parser.add_argument("--movie-max-frames", type=int, default=161)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("CPU thread count must be positive")
    if args.movie_fps < 1 or args.movie_max_frames < 2:
        parser.error("Movie FPS must be positive and max frames at least two.")
    return args


if __name__ == "__main__":
    args = parse_args()
    # ---- Notebook cell 1 ----
    from pathlib import Path
    from functools import lru_cache
    import hashlib
    import json
    import math
    import marshal
    import sys
    import time
    import numpy as np
    from html import escape
    import torch
    from scipy.integrate import solve_bvp, cumulative_trapezoid
    from scipy.optimize import root, brentq
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from tqdm.auto import tqdm

    KNEEP_ROOT = Path(__file__).resolve().parents[1]
    LRAMB_ROOT = args.lramb_root.resolve() if args.lramb_root else KNEEP_ROOT.parent / "LRAMB"
    sys.path.insert(0, str(LRAMB_ROOT))
    from lramb.model import KernelSpec
    from lramb import stencils as stencil_ops
    from lramb.torch_stencils import RealSpaceStepper
    from lramb.circle import cluster_observables

    DEVICE = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if DEVICE.type == "cuda":
        torch.cuda.set_device(DEVICE)
    torch.set_num_threads(args.cpu_threads)
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                         "grid.alpha": 0.2, "animation.embed_limit": 100})
    def display_table(rows):
        print(json.dumps(list(rows), ensure_ascii=False, indent=2, default=str))

    figure_number = 0
    def save_figures():
        global figure_number
        for number in plt.get_fignums():
            figure_number += 1
            fig = plt.figure(number)
            path = FIGURE_ROOT / f"figure_{figure_number:03d}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Figure:", path)

    print("Device:", DEVICE)


    # ---- Notebook cell 3 ----
    GRID = dict(n=64, dx=1.0)
    BASE = dict(a=-0.25, b=0.25, kappa=4.0, mobility=1.0,
                g=0.0, ell=1.0, lam=10.0, zeta=0.0, D=0.1, radius=16.0)
    # Matches Corr_AMB2D_ShellForce; KNEEP uses the opposite sign for a.
    CASES = [
        dict(name="target"),
        # Optional comparisons (uncomment as needed):
        # dict(name="local passive", g=0.0, lam=0.0),
        # dict(name="Poisson passive", g=0.3, lam=0.0),
        # dict(name="Poisson active", g=0.3),
        # dict(name="target without noise", D=0.0),
        # dict(name="smaller droplet", radius=12.0),
        # dict(name="larger droplet", radius=20.0),
    ]
    INITIALIZATION = "tanh"  # "tanh", "local_bvp", or "periodic_stationary"
    REMOVE_ZERO_MODE = True
    BVP = dict(nodes=400, tol=1e-5, max_nodes=12000)
    ROOT = dict(tol=1e-7, maxiter=100, continuation_steps=8)
    # 10,000 burn steps; 4,999 production transitions match 5,000 test frames.
    RUN = dict(dt=0.001, burn_in=10.0, duration=4.999,
               save_every=0.1, ep_blocks=8, seed=123)
    MODE_MAX = 8
    RULES = dict(mass_tolerance=1e-10)
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        groups = dict(GRID=GRID, BASE=BASE, RUN=RUN, BVP=BVP, ROOT=ROOT, RULES=RULES)
        allowed = set(groups) | {"CASES", "INITIALIZATION", "REMOVE_ZERO_MODE", "MODE_MAX"}
        if not isinstance(overrides, dict) or set(overrides)-allowed:
            raise ValueError(f"Config keys must be in {sorted(allowed)}")
        for name, target in groups.items():
            if name in overrides:
                value = overrides[name]
                if not isinstance(value, dict) or set(value)-set(target):
                    raise ValueError(f"Invalid {name} keys; expected {sorted(target)}")
                target.update(value)
        CASES = overrides.get("CASES", CASES)
        INITIALIZATION = overrides.get("INITIALIZATION", INITIALIZATION)
        REMOVE_ZERO_MODE = overrides.get("REMOVE_ZERO_MODE", REMOVE_ZERO_MODE)
        MODE_MAX = overrides.get("MODE_MAX", MODE_MAX)
    CASES = [{**BASE, **case} for case in CASES]
    OUTPUT_ROOT = args.output_dir.resolve() if args.output_dir else KNEEP_ROOT / "results" / "poisson_sanity"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT = OUTPUT_ROOT / ("figures_" + time.strftime("%Y%m%d_%H%M%S"))
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    NUMERICS_VERSION = "tjhung-realspace-mixed-mobility-ep-v3"

    n, dx = GRID["n"], GRID["dx"]
    length, dV = n*dx, dx**2
    outer_radius = length/2
    if n < 16 or n % 2 or dx <= 0:
        raise ValueError("Use positive dx and even n >= 16.")
    if INITIALIZATION not in ("periodic_stationary", "local_bvp", "tanh"):
        raise ValueError("Choose periodic_stationary, local_bvp, or tanh.")
    if not CASES or len({c["name"] for c in CASES}) != len(CASES):
        raise ValueError("Provide cases with unique names.")
    for c in CASES:
        if (not np.isfinite([v for k,v in c.items() if k != "name"]).all()
            or c["a"] >= 0 or min(c["b"], c["kappa"], c["ell"], c["mobility"]) <= 0
            or c["g"] < 0 or c["D"] < 0 or not 3*dx < c["radius"] < length/3):
            raise ValueError(f"Invalid double well, kernel, noise, or droplet radius: {c}")
    if not all(np.isfinite(RUN[k]) for k in ("dt", "burn_in", "duration", "save_every")):
        raise ValueError("Run times must be finite.")
    if min(RUN["dt"], RUN["duration"], RUN["save_every"]) <= 0 or RUN["burn_in"] < 0:
        raise ValueError("Use nonnegative burn_in and positive dt/duration/save_every.")
    burn_steps = round(RUN["burn_in"]/RUN["dt"])
    production_steps = round(RUN["duration"]/RUN["dt"])
    stride = round(RUN["save_every"]/RUN["dt"])
    for count, key in ((burn_steps,"burn_in"), (production_steps,"duration"), (stride,"save_every")):
        if not np.isclose(count*RUN["dt"], RUN[key], rtol=0, atol=1e-10):
            raise ValueError(f"{key} must be an integer multiple of dt.")
    if stride < 1 or production_steps < 1:
        raise ValueError("Need at least one production step and a save stride >= 1.")
    if not isinstance(RUN["ep_blocks"], int) or not 2 <= RUN["ep_blocks"] <= production_steps:
        raise ValueError("ep_blocks must be an integer between 2 and production_steps.")
    if not isinstance(MODE_MAX, int) or not 1 <= MODE_MAX < n//2:
        raise ValueError("MODE_MAX must be an integer in [1, n/2).")
    steps = burn_steps + production_steps
    MODE_NUMBERS = np.arange(1, MODE_MAX+1)
    EP_CHANNELS = ("local conservative", "Poisson conservative", "active", "medium total")
    display_table(CASES)
    print(f"Burn-in: 0..{RUN['burn_in']:g}; production: {RUN['burn_in']:g}..{steps*RUN['dt']:g}")
    print("All diagnostics include burn-in; EP maps exclude it. D=0 EP is undefined.")
    print("Output:", OUTPUT_ROOT)


    # ---- Notebook cell 5 ----
    @lru_cache(maxsize=32)
    def solve_radial_bvp(a, b, kappa, nominal_radius, rb, nodes, tol, max_nodes):
        phi0 = math.sqrt(-a/b)
        width = math.sqrt(2*kappa/(-a))
        mean_disk = phi0*(2*(nominal_radius/rb)**2 - 1)
        target_integral = mean_disk*rb**2/2
        r = np.linspace(0, rb, nodes)
        S = np.zeros((3,3))
        S[1,1] = -1.0

        def equation(r, y, p):
            phi, derivative, integral = y
            return np.vstack((derivative, (a*phi+b*phi**3-p[0])/kappa, r*phi))

        def boundary(left, right, p):
            return np.array([left[1], right[1], left[2],
                             (right[2]-target_integral)/rb**2])

        failures = []
        for factor in (1.0, 0.9, 1.1):
            z = (factor*nominal_radius-r)/width
            guess_phi = phi0*np.tanh(z)
            guess_derivative = -phi0/width*(1-np.tanh(z)**2)
            guess = np.vstack((guess_phi, guess_derivative,
                               cumulative_trapezoid(r*guess_phi, r, initial=0)))
            mu_guess = kappa*phi0/(nominal_radius*width)
            solution = solve_bvp(equation, boundary, r, guess, p=[mu_guess], S=S,
                                 tol=tol, bc_tol=tol, max_nodes=max_nodes)
            dense_r = np.linspace(0, rb, 2001)
            values = solution.sol(dense_r)
            phi = values[0]
            bc_error = float(np.max(np.abs(boundary(solution.y[:,0], solution.y[:,-1], solution.p))))
            rms = float(np.max(solution.rms_residuals))
            droplet_branch = (np.isfinite(values).all() and phi[0] > 0 and phi[-1] < 0
                              and phi[0]-phi[-1] > phi0
                              and np.max(np.diff(phi)) < 1e-3*phi0)
            if solution.success and bc_error < 10*tol and rms < 10*tol and droplet_branch:
                radius = brentq(lambda x: float(solution.sol(x)[0]), 0, rb)
                info = dict(mu=float(solution.p[0]), center=float(phi[0]), exterior=float(phi[-1]),
                            nominal_radius=nominal_radius, crossing_radius=radius,
                            disk_mean=mean_disk, bc_error=bc_error, max_rms_residual=rms,
                            nodes=int(solution.x.size))
                return solution, info
            failures.append(f"status={solution.status}, branch={droplet_branch}, residual={rms:.3g}")
        raise RuntimeError("No accepted droplet BVP branch: " + "; ".join(failures))

    coord = (np.arange(n)*dx-length/2+length/2) % length-length/2
    X, Y = np.meshgrid(coord, coord)
    DISTANCE = np.hypot(X, Y)
    THETA = np.arctan2(Y, X)

    def local_bvp_field(case):
        sol, info = solve_radial_bvp(case["a"], case["b"], case["kappa"],
            case["radius"], outer_radius, BVP["nodes"], BVP["tol"], BVP["max_nodes"])
        field = sol.sol(np.minimum(DISTANCE, outer_radius).ravel())[0].reshape(n,n)
        return field.copy(), dict(info, square_mean=float(field.mean())), sol

    def tanh_field(case):
        amplitude = math.sqrt(-case["a"]/case["b"])
        width = math.sqrt(2*case["kappa"]/(-case["a"]))
        field = amplitude*np.tanh((case["radius"]-DISTANCE)/width)
        info = dict(method="tanh", nominal_radius=case["radius"],
                    amplitude=amplitude, interface_width=width, square_mean=float(field.mean()))
        return field, info

    # Editable choice for visualizing the boundary solution.
    INITIAL_CASE = next(c for c in CASES if c["name"] == "target") if any(c["name"] == "target" for c in CASES) else CASES[0]
    try:
        r = np.linspace(0, outer_radius, 1000)
        if INITIALIZATION == "tanh":
            bvp_field, bvp_info = tanh_field(INITIAL_CASE)
            radial_profile = bvp_info["amplitude"]*np.tanh(
                (INITIAL_CASE["radius"]-r)/bvp_info["interface_width"])
            preview_label = "Analytic tanh profile"
        else:
            bvp_field, bvp_info, bvp_solution = local_bvp_field(INITIAL_CASE)
            radial_profile = bvp_solution.sol(r)[0]
            preview_label = "Radial BVP solution"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(r, radial_profile, label=preview_label)
        phi0 = math.sqrt(-INITIAL_CASE["a"]/INITIAL_CASE["b"])
        axes[0].axhline(phi0, color="gray", ls=":", label="local minima")
        axes[0].axhline(-phi0, color="gray", ls=":")
        axes[0].axvline(INITIAL_CASE["radius"], color="black", ls="--", label="nominal radius")
        axes[0].set(xlabel="r", ylabel="phi", title=preview_label)
        axes[0].legend()
        im = axes[1].imshow(bvp_field, origin="lower", extent=(0,length,0,length), cmap="RdBu_r")
        axes[1].set(title=preview_label, xlabel="x", ylabel="y")
        axes[1].grid(False)
        fig.colorbar(im, ax=axes[1], label="phi")
        fig.tight_layout()
        save_figures()
        display_table([bvp_info])
    except (RuntimeError, ValueError) as error:
        print("Initial profile preview failed:", error)
        print("The sweep still attempts all cases; failure is not a stability verdict.")


    # ---- Notebook cell 7 ----
    wave = 2*np.pi*np.fft.fftfreq(n, d=dx)
    bond = 2*np.sin(wave*dx/2)/dx
    QX, QY = np.meshgrid(bond, bond)
    Q2 = QX**2 + QY**2

    def kernel_multiplier(case):
        multiplier = KernelSpec(kind="poisson", ell=case["ell"]).multiplier(np.sqrt(Q2))
        multiplier[0, 0] = 0.0 if REMOVE_ZERO_MODE else 1.0
        return multiplier

    def laplacian_numpy(phi):
        return stencil_ops.laplacian9(phi,dx)

    def grad2_numpy(phi):
        return stencil_ops.gradient_squared(phi,dx)

    def chemical_potential(phi, case):
        convolution = np.fft.ifft2(kernel_multiplier(case)*np.fft.fft2(phi)).real if case["g"] else 0.
        return (case["a"]*phi+case["b"]*phi**3-case["kappa"]*laplacian_numpy(phi)
                +case["g"]*convolution+stencil_ops.effective_active(phi,dx,case["lam"],case.get("zeta",0.)))

    def residual_report(phi, case):
        mu=chemical_potential(phi,case)
        drift=case["mobility"]*stencil_ops.mobility8(mu,dx)
        return dict(mu=float(mu.mean()),mu_spread=float(np.max(abs(mu-stencil_ops.null_projection(mu)))),
                    drift_rms=float(np.sqrt(np.mean(drift**2))),square_mean=float(phi.mean()))

    # Reuse deterministic initial states for cases differing only by noise or name.
    INITIAL_STATES = {}

    def make_initial_state(case):
        key = json.dumps(dict(physics={k:v for k,v in case.items() if k not in ("name", "D")},
                              grid=GRID, bvp=BVP, root=ROOT, initialization=INITIALIZATION,
                              zero_mode=REMOVE_ZERO_MODE), sort_keys=True)
        if key in INITIAL_STATES:
            field, info = INITIAL_STATES[key]
            return field.copy(), dict(info)
        if INITIALIZATION == "tanh":
            field, radial_info = tanh_field(case)
        else:
            field, radial_info, _ = local_bvp_field(case)
        target_mean = float(field.mean())
        reference_field = field.copy()
        before = residual_report(field, case)
        continuation = []
        if INITIALIZATION == "periodic_stationary":
            previous_mu = float(chemical_potential(field, {**case, "g":0., "lam":0., "zeta":0.}).mean())
            fractions = np.linspace(0., 1., ROOT["continuation_steps"]+1)
            if case["g"] == 0 and case["lam"] == 0 and case.get("zeta",0.) == 0:
                fractions = [0.]
            for fraction in fractions:
                print(f"  stationary continuation {fraction:.2f}", flush=True)
                current = {**case, "g":fraction*case["g"], "lam":fraction*case["lam"], "zeta":fraction*case.get("zeta",0.)}
                def equation(z):
                    phi = z[:-1].reshape(n, n)
                    return np.r_[(stencil_ops.stationary_residual(phi,reference_field,chemical_potential(phi,current))+z[-1]).ravel(),
                                 n*n*(phi.mean()-target_mean)]
                solved = root(equation, np.r_[field.ravel(), previous_mu], method="krylov",
                              options=dict(fatol=ROOT["tol"], maxiter=ROOT["maxiter"]))
                error = float(np.max(abs(equation(solved.x))))
                if not solved.success or not np.isfinite(error) or error > 5*ROOT["tol"]:
                    raise RuntimeError(f"Periodic initialization failed at continuation={fraction:.3f}: "
                                       f"residual={error:.3g}; {solved.message}")
                field = solved.x[:-1].reshape(n, n)
                previous_mu = float(solved.x[-1])
                morphology = cluster_observables(field[None], dx, threshold=0., modes=2)
                phi0 = np.sqrt(-case["a"]/case["b"])
                if (field[n//2,n//2] <= 0 or field[0,0] >= 0
                    or field.max()-field.min() < phi0
                    or morphology["components"][0] != 1
                    or morphology["coverage"][0] < 1):
                    raise RuntimeError(f"Continuation={fraction:.3f} left the resolved droplet branch.")
                continuation.append(dict(fraction=float(fraction), residual=error,
                                         iterations=int(getattr(solved, "nit", -1))))
        after = residual_report(field, case)
        if abs(field.mean()-target_mean) > RULES["mass_tolerance"]:
            raise RuntimeError("Stationary initialization did not preserve the target square mean.")
        info = dict(radial=radial_info, before=before, after=after,
                    continuation=continuation, initialization=INITIALIZATION)
        INITIAL_STATES[key] = (field.copy(), info)
        return field, info


    # ---- Notebook cell 9 ----
    try:
        preview, preview_info = make_initial_state(INITIAL_CASE)
        display_table([dict(stage="tanh profile" if INITIALIZATION == "tanh" else "embedded local BVP", **preview_info["before"]),
                       dict(stage="chosen initialization", **preview_info["after"])])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, field, title in zip(axes, [preview, chemical_potential(preview, INITIAL_CASE)],
                                    ["Initial field", "Initial chemical potential"]):
            image = ax.imshow(field, origin="lower", extent=[0,length,0,length], cmap="coolwarm")
            ax.set(title=title, xlabel="x", ylabel="y")
            ax.grid(False)
            fig.colorbar(image, ax=ax)
        save_figures()
    except (RuntimeError, ValueError) as error:
        print("Target initialization unavailable:", error)
        print("The sweep below still attempts the other cases.")


    # ---- Notebook cell 11 ----
    @torch.no_grad()
    def evolve(initial, case, progress=True):
        dtype = torch.float64
        phi = torch.as_tensor(initial.copy(), dtype=dtype, device=DEVICE)
        q2 = torch.as_tensor(stencil_ops.symbols(n,dx)[0], dtype=dtype, device=DEVICE)
        q = torch.as_tensor(np.sqrt(Q2), dtype=dtype, device=DEVICE)
        kernel = torch.as_tensor(kernel_multiplier(case), dtype=dtype, device=DEVICE)
        stepper = RealSpaceStepper(n,dx,case["a"],case["b"],case["kappa"],case["lam"],
            case["mobility"],case["D"],RUN["dt"],kernel,case["g"],case.get("zeta",0.),DEVICE,dtype)
        rng = torch.Generator(device=DEVICE).manual_seed(RUN["seed"])
        initial_mean = phi.mean().clone()
        phi_hat = torch.fft.fft2(phi, norm="ortho")
        ep_defined = case["D"] > 0
        maps = torch.zeros((4,n,n), dtype=dtype, device=DEVICE)
        block_maps = torch.zeros((RUN["ep_blocks"],4,n,n), dtype=dtype, device=DEVICE)
        block_steps = np.zeros(RUN["ep_blocks"], dtype=np.int64)
        interval_entropy = torch.zeros(4, dtype=dtype, device=DEVICE)
        interval_start = 0
        state = {key: [] for key in ("fields", "time", "mass_mean", "variance", "energy",
                                    "mean_q", "mu_spread", "drift_rms", "axial_modes")}
        ep_time, ep_duration, ep_phase, ep_increment = [], [], [], []

        def potentials(field, field_hat):
            return stepper.potentials(field)

        def record(step):
            nonlocal interval_start
            power = phi_hat.abs().square()
            connected = power.clone()
            connected[0,0] = 0
            energy = torch.stack((dV*(case["a"]/2*phi.square()+case["b"]/4*phi.pow(4)).sum(),
                dV*case["kappa"]/2*(q2*power).sum(),
                dV*case["g"]/2*(kernel*power).sum()))
            mu = potentials(phi, phi_hat).sum(0)
            drift = stepper.drift(phi)
            # Array axes are [y,x]. Fourier coefficients are normalized by n*n.
            modes = torch.stack((phi_hat[0,1:MODE_MAX+1], phi_hat[1:MODE_MAX+1,0]))/n
            state["fields"].append(phi.float().cpu().numpy().copy())
            state["time"].append(step*RUN["dt"])
            for key, value in (("mass_mean",phi.mean()), ("variance",phi.var(correction=0)),
                ("energy",energy), ("mean_q",(q*connected).sum()/connected.sum().clamp_min(1e-30)),
                ("mu_spread",(mu-stepper.null_projection(mu)).abs().max()), ("drift_rms",drift.square().mean().sqrt()),
                ("axial_modes",modes)):
                state[key].append(value.cpu().numpy().copy())
            if step > interval_start:
                elapsed = (step-interval_start)*RUN["dt"]
                ep_time.append((step+interval_start)*RUN["dt"]/2)
                ep_duration.append(elapsed)
                ep_phase.append("burn-in" if step <= burn_steps else "production")
                ep_increment.append(interval_entropy.cpu().numpy().copy() if ep_defined else np.full(4,np.nan))
                interval_entropy.zero_()
                interval_start = step

        record(0)
        for step in tqdm(range(1,steps+1), desc=case["name"], leave=False, disable=not progress):
            normal = torch.randn((2,n,n),generator=rng,device=DEVICE,dtype=dtype)
            new = stepper(phi,normal)
            next_hat = torch.fft.fft2(new,norm="ortho")
            if ep_defined:
                channels = -potentials((phi+new)/2,(phi_hat+next_hat)/2)*(new-phi)/case["D"]
                ds = torch.cat((channels,channels.sum(0,keepdim=True)),dim=0)
                interval_entropy += dV*ds.sum(dim=(-2,-1))
                if step > burn_steps:
                    block = (step-burn_steps-1)*RUN["ep_blocks"]//production_steps
                    maps += ds
                    block_maps[block] += ds
            if step > burn_steps:
                block_steps[(step-burn_steps-1)*RUN["ep_blocks"]//production_steps] += 1
            phi, phi_hat = new, next_hat
            if step % 100 == 0 or step in (burn_steps,steps):
                if not bool(torch.isfinite(phi).all()) or float(phi.abs().max()) > 30:
                    raise FloatingPointError(f"Unstable Euler state at t={step*RUN['dt']}; decrease dt.")
                if float((phi.mean()-initial_mean).abs()) > RULES["mass_tolerance"]:
                    raise FloatingPointError("Mass conservation tolerance exceeded.")
            # Force a record at burn boundary to keep diagnostic EP intervals disjoint.
            if step % stride == 0 or step in (burn_steps,steps):
                record(step)

        result = {key: np.asarray(values) for key,values in state.items()}
        duration = production_steps*RUN["dt"]
        block_duration = block_steps*RUN["dt"]
        result.update(ep_time=np.asarray(ep_time), ep_duration=np.asarray(ep_duration),
            ep_phase=np.asarray(ep_phase), ep_increment=np.asarray(ep_increment),
            ep_rate=np.asarray(ep_increment)/np.asarray(ep_duration)[:,None],
            ep_map=(maps/duration).cpu().numpy() if ep_defined else np.full((4,n,n),np.nan),
            ep_map_blocks=(block_maps/torch.as_tensor(block_duration,device=DEVICE)[:,None,None,None]).cpu().numpy()
                          if ep_defined else np.full((RUN["ep_blocks"],4,n,n),np.nan),
            ep_block_duration=block_duration, ep_defined=np.asarray(ep_defined))
        if ep_defined:
            production = result["ep_phase"] == "production"
            np.testing.assert_allclose(result["ep_duration"][production].sum(),duration,rtol=1e-12)
            np.testing.assert_allclose(dV*result["ep_map"].sum((-2,-1)),
                result["ep_increment"][production].sum(0)/duration,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(result["ep_map"][:3].sum(0),result["ep_map"][3],atol=1e-9)
        return result

    def cache_description(case, initial):
        description = dict(version=NUMERICS_VERSION, case=case, grid=GRID, run=RUN,
            initialization=INITIALIZATION, bvp=BVP, root=ROOT, mode_max=MODE_MAX,
            remove_zero_mode=REMOVE_ZERO_MODE, initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
            multiplier_sha256=hashlib.sha256(kernel_multiplier(case).tobytes()).hexdigest(),
            solver_sha256=hashlib.sha256(marshal.dumps(evolve.__wrapped__.__code__)+b"".join((LRAMB_ROOT / "lramb" / file).read_bytes() for file in ("stencils.py","torch_stencils.py"))).hexdigest(),
            device=str(DEVICE), torch_version=torch.__version__)
        return description, hashlib.sha256(json.dumps(description,sort_keys=True).encode()).hexdigest()[:20]

    RESULTS, FAILURES = {}, []
    for case in CASES:
        stage = "initialization"
        try:
            print(f"Preparing {case['name']}: g={case['g']}, ell={case['ell']}, lambda={case['lam']}, D={case['D']}",flush=True)
            initial, init_info = make_initial_state(case)
            description, signature = cache_description(case,initial)
            path = OUTPUT_ROOT/f"trajectory_{signature}.npz"
            stage = "simulation/cache"
            if path.exists():
                with np.load(path,allow_pickle=False) as saved:
                    trajectory = {key:saved[key].copy() for key in saved.files if key != "metadata"}
                print("Loaded",path.name)
            else:
                trajectory = evolve(initial,case)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle,**trajectory,
                        metadata=json.dumps(dict(description=description,initialization=init_info)))
                temporary.replace(path)
            RESULTS[case["name"]] = dict(case=case,initial=initial,initialization=init_info,
                                         trajectory=trajectory,path=path)
        except (RuntimeError,ValueError,FloatingPointError) as error:
            FAILURES.append(dict(name=case["name"],stage=stage,message=str(error)))
            print(f"{stage} FAILED for {case['name']}: {error}")
    if FAILURES:
        display_table(FAILURES)
    print(f"Completed {len(RESULTS)}/{len(CASES)} cases; diagnostics cover t=0 through production end.")


    # ---- Notebook cell 13 ----
    PLOT_NAMES = [name for name in ("local passive","local active","Poisson passive","target","target with noise")
                  if name in RESULTS]
    # PLOT_NAMES = [name for name in ("shorter ell=0.5","target","longer ell=2") if name in RESULTS]
    if not PLOT_NAMES and RESULTS:
        PLOT_NAMES = [next(iter(RESULTS))]

    def mark_phases(ax, time_on_y=False):
        boundary = RUN["burn_in"]
        if time_on_y:
            ax.axhspan(0,boundary,color="gray",alpha=.10)
            ax.axhline(boundary,color="black",ls="--",lw=1)
        else:
            ax.axvspan(0,boundary,color="gray",alpha=.10)
            ax.axvline(boundary,color="black",ls="--",lw=1,label="production starts")

    if PLOT_NAMES:
        bound = max(float(np.abs(RESULTS[name]["trajectory"]["fields"]).max()) for name in PLOT_NAMES)
        fig,axes = plt.subplots(len(PLOT_NAMES),3,figsize=(12,3.2*len(PLOT_NAMES)),squeeze=False,constrained_layout=True)
        for row,name in enumerate(PLOT_NAMES):
            tr = RESULTS[name]["trajectory"]
            indices = [0,int(np.argmin(abs(tr["time"]-RUN["burn_in"]))),len(tr["time"])-1]
            for ax,index,label in zip(axes[row],indices,("initial","production start","production end")):
                image = ax.imshow(tr["fields"][index],origin="lower",extent=(0,length,0,length),
                                  cmap="RdBu_r",vmin=-bound,vmax=bound)
                ax.set(title=f"{name}: {label}, t={tr['time'][index]:g}",xlabel="x",ylabel="y")
                ax.grid(False)
        fig.colorbar(image,ax=axes.ravel().tolist(),label="phi",shrink=.8)
        save_figures()


    # ---- Notebook cell 14 ----
    SELECTED_MODES = [m for m in (1,2,4,8) if m <= MODE_MAX]
    for name in PLOT_NAMES:
        tr = RESULTS[name]["trajectory"]
        t,fields = tr["time"],tr["fields"]
        power = np.abs(tr["axial_modes"])**2
        positive_power = power[power>0]
        floor = max(float(positive_power.max())*1e-12,1e-30) if positive_power.size else 1e-30
        log_power = np.log10(np.maximum(power,floor))
        fig,axes = plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
        profile_bound = float(abs(fields).max())
        for axis,label in enumerate(("x","y")):
            profile = fields.mean(axis=1 if axis == 0 else 2)
            image = axes[axis,0].pcolormesh(np.arange(n)*dx,t,profile,shading="auto",
                cmap="RdBu_r",vmin=-profile_bound,vmax=profile_bound)
            axes[axis,0].set(xlabel=label,ylabel="absolute time",title=f"Mean {label} profile")
            fig.colorbar(image,ax=axes[axis,0],label="phi")
            image = axes[axis,1].pcolormesh(MODE_NUMBERS,t,log_power[:,axis],shading="auto",
                cmap="magma",vmin=float(log_power.min()),vmax=float(log_power.max()))
            axes[axis,1].set(xlabel=f"{label} mode number m",ylabel="absolute time",title=f"Axial {label} mode power")
            fig.colorbar(image,ax=axes[axis,1],label="log10 |a_m|^2")
            for m in SELECTED_MODES:
                axes[axis,2].plot(t,np.abs(tr["axial_modes"][:,axis,m-1]),label=f"m={m}")
            axes[axis,2].set(xlabel="absolute time",ylabel="|a_m|",title=f"{label} mode amplitudes")
            axes[axis,2].legend(fontsize=8)
            for ax in axes[axis,:2]:
                mark_phases(ax,time_on_y=True)
                ax.grid(False)
            mark_phases(axes[axis,2])
        fig.suptitle(f"{name}: x/y modes from initialization; dashed line = production starts")
        save_figures()


    # ---- Notebook cell 16 ----
    SUMMARY = []
    for name in PLOT_NAMES:
        tr = RESULTS[name]["trajectory"]
        t = tr["time"]
        fig,axes = plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
        axes[0,0].plot(t,tr["mass_mean"]-tr["mass_mean"][0])
        axes[0,0].set(title="Conserved mean drift",ylabel="mean phi - initial mean")
        axes[0,1].plot(t,tr["variance"])
        axes[0,1].set(title="Field variance",ylabel="spatial variance")
        for j,label in enumerate(EP_CHANNELS[:2]):
            values = tr["energy"][:,:2].sum(1) if j == 0 else tr["energy"][:,2]
            axes[0,2].plot(t,values/(length**2),label=label)
        axes[0,2].plot(t,tr["energy"].sum(1)/(length**2),color="black",ls="--",label="total conservative")
        axes[0,2].set(title="Conservative energy density",ylabel="F / area")
        axes[0,2].legend(fontsize=8)
        axes[1,0].plot(t,tr["mean_q"])
        axes[1,0].set(title="Connected structure mean q",ylabel="mean q")
        axes[1,1].plot(t,tr["drift_rms"])
        axes[1,1].set(title="Deterministic drift RMS",ylabel="RMS(M Laplacian mu)")
        if bool(tr["ep_defined"]):
            axes[1,2].plot(tr["ep_time"],tr["ep_rate"][:,3],alpha=.6,label="medium total")
            axes[1,2].plot(tr["ep_time"],tr["ep_rate"][:,2],alpha=.8,label="active")
            axes[1,2].legend(fontsize=8)
        else:
            axes[1,2].text(.5,.5,"D=0: entropy / D undefined",ha="center",transform=axes[1,2].transAxes)
        axes[1,2].set(title="Full-step interval EP rates",ylabel="global EP / time")
        for ax in axes.flat:
            mark_phases(ax)
            ax.set(xlabel="absolute time (including burn-in)", xlim=(0, steps*RUN["dt"]))
        fig.suptitle(name)
        save_figures()
        production = t > RUN["burn_in"]
        values = tr["variance"][production]
        halves = np.array_split(values,2)
        SUMMARY.append(dict(name=name,burn_in=RUN["burn_in"],production_time=RUN["duration"],
            max_mass_drift=float(np.max(abs(tr["mass_mean"]-tr["mass_mean"][0]))),
            production_variance_first_half=float(halves[0].mean()) if len(halves[0]) else None,
            production_variance_second_half=float(halves[1].mean()) if len(halves[1]) else None,
            medium_mean_rate=float(dV*tr["ep_map"][3].sum()) if bool(tr["ep_defined"]) else None))
    display_table(SUMMARY)

    NOISY_NAMES = [name for name in PLOT_NAMES if bool(RESULTS[name]["trajectory"]["ep_defined"])]
    if NOISY_NAMES:
        fig,axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for name in NOISY_NAMES:
            tr = RESULTS[name]["trajectory"]
            block_rate = dV*tr["ep_map_blocks"].sum((-2,-1))
            block_time = RUN["burn_in"]+np.cumsum(tr["ep_block_duration"])-tr["ep_block_duration"]/2
            axes[0].plot(block_time,block_rate[:,3],"o-",label=name)
            axes[1].plot(block_time,block_rate[:,2],"o-",label=name)
        for ax,title in zip(axes,("Production medium EP block means","Production active EP block means")):
            ax.set(title=title,xlabel="absolute time after burn-in",ylabel="global EP / time")
            ax.legend(fontsize=8)
        save_figures()


    # ---- Notebook cell 18 ----
    EP_PLOT_NAMES = [name for name in PLOT_NAMES if bool(RESULTS[name]["trajectory"]["ep_defined"])]
    # EP_PLOT_NAMES = [name for name,r in RESULTS.items() if bool(r["trajectory"]["ep_defined"])]
    if EP_PLOT_NAMES:
        limits = np.maximum(np.max([abs(RESULTS[name]["trajectory"]["ep_map"]).max(axis=(-2,-1))
                                   for name in EP_PLOT_NAMES],axis=0),1e-12)
        fig,axes = plt.subplots(len(EP_PLOT_NAMES),4,figsize=(16,3.5*len(EP_PLOT_NAMES)),
                                squeeze=False,constrained_layout=True)
        for row,name in enumerate(EP_PLOT_NAMES):
            tr = RESULTS[name]["trajectory"]
            production = tr["ep_phase"] == "production"
            map_integrals = dV*tr["ep_map"].sum((-2,-1))
            interval_means = tr["ep_increment"][production].sum(0)/tr["ep_duration"][production].sum()
            np.testing.assert_allclose(map_integrals,interval_means,rtol=1e-9,atol=1e-9)
            for j,(ax,channel) in enumerate(zip(axes[row],EP_CHANNELS)):
                image = ax.imshow(tr["ep_map"][j],origin="lower",extent=(0,length,0,length),
                                  cmap="RdBu_r",vmin=-limits[j],vmax=limits[j])
                ax.set(title=f"{name}: {channel}\nintegral={map_integrals[j]:.4g}",xlabel="x",ylabel="y")
                ax.grid(False)
                fig.colorbar(image,ax=ax,label="EP density / time",shrink=.8)
        fig.suptitle(f"Production-only EP maps: t={RUN['burn_in']:g}..{steps*RUN['dt']:g}")
        save_figures()
    else:
        print("No D>0 case selected. Set BASE['D'] > 0 or select a noisy case in EP_PLOT_NAMES.")


    # ---- Notebook cell 20 ----
    INSPECT_NAME = next((name for name in ("target with noise","target") if name in RESULTS),next(iter(RESULTS),None))
    MAKE_MOVIE = args.movie != "none"
    MOVIE_FPS = args.movie_fps
    MAX_MOVIE_FRAMES = args.movie_max_frames
    if MAKE_MOVIE and INSPECT_NAME is not None:
        tr = RESULTS[INSPECT_NAME]["trajectory"]
        fields,t = tr["fields"],tr["time"]
        selected = np.unique(np.r_[np.linspace(0,len(t)-1,min(MAX_MOVIE_FRAMES,len(t))).astype(int),
                                    np.argmin(abs(t-RUN["burn_in"]))])
        bound = float(abs(fields).max())
        coord = np.arange(n)*dx
        fig,axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        image = axes[0].imshow(fields[0],origin="lower",extent=(0,length,0,length),
                               cmap="RdBu_r",vmin=-bound,vmax=bound)
        axes[0].set(title="Field",xlabel="x",ylabel="y")
        axes[0].grid(False)
        fig.colorbar(image,ax=axes[0],shrink=.8,label="phi")
        # fields have axes [time, y, x]: fix x=L/2 and plot phi against y.
        center_x = n//2
        axes[0].axvline(center_x*dx, color="black", ls="--", lw=1)
        centerline, = axes[1].plot(coord, fields[0, :, center_x])
        axes[1].set(xlabel="y", ylabel="phi", ylim=(-bound,bound),
                    title=f"Central cut at x={center_x*dx:g}")
        title = fig.suptitle("")
        def update_movie(frame):
            index = int(selected[frame])
            image.set_data(fields[index])
            centerline.set_ydata(fields[index, :, center_x])
            phase = "burn-in" if t[index] < RUN["burn_in"] else "production"
            title.set_text(f"{INSPECT_NAME} | t={t[index]:.2f} | {phase}")
            return image,centerline,title
        profile_movie = animation.FuncAnimation(fig,update_movie,frames=len(selected),interval=1000/MOVIE_FPS)
        plt.close(fig)
        if args.movie == "mp4":
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError("MP4 requires ffmpeg. Use --movie html instead.")
            movie_path = FIGURE_ROOT / "central_x_cut.mp4"
            profile_movie.save(movie_path, fps=MOVIE_FPS)
        else:
            movie_path = FIGURE_ROOT / "central_x_cut.html"
            movie_path.write_text("<!doctype html><meta charset='utf-8'>" +
                                  profile_movie.to_jshtml(fps=MOVIE_FPS), encoding="utf-8")
        print("Movie:", movie_path)


    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} case(s) failed; see diagnostics above.")
