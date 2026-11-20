"""Batched g sweep: independent replicas, active-only EP and stationary movies.

Run from any directory; see --help. Requires the sibling LRAMB repository.
All g/replica pairs share one vectorized solver call, with independent noise.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.lramb_poisson import _lramb_module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--g', nargs='+', type=float, default=[0, .25, .5, 1, 2])
    p.add_argument('--trajectories', type=int, default=8)
    p.add_argument('--burn-time', type=float, default=20000)
    p.add_argument('--production-time', type=float, default=1000)
    p.add_argument('--dt', type=float, default=.01)
    p.add_argument('--diagnostic-every', type=int, default=1000,
                   help='steps between field variance records, including burn-in')
    p.add_argument('--movie-every', type=int, default=500,
                   help='production steps between movie/cluster samples')
    p.add_argument('--graph-steps', type=int, default=20,
                   help='CUDA graph chunk length; 0 disables, CPU always eager')
    p.add_argument('--device', default='auto')
    p.add_argument('--cpu-threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--n', type=int, default=32)
    for name, default in dict(dx=1., a=-1., b=1., kappa=1., lam=.8,
                              zeta=0., mobility=1., D=.05, mean=-.2, ell=2.).items():
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--initial', choices=['random', 'stripe'], default='random')
    p.add_argument('--cluster-threshold', type=float, default=0.)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--movie-format', choices=['gif', 'mp4'], default='gif')
    p.add_argument('--lramb-root', type=Path)
    p.add_argument('--output', type=Path, default=ROOT/'results'/'g_steady')
    p.add_argument('--plot-only', action='store_true', help='render existing saved data')
    return p.parse_args()


def step_count(duration, dt):
    ratio = duration / dt
    if not math.isfinite(ratio) or ratio < 0 or not math.isclose(ratio, round(ratio), abs_tol=1e-7):
        raise ValueError('Times must be nonnegative integer multiples of dt')
    return round(ratio)


class Chunk:
    """Fixed-address graph with advancing custom RNG and float64 EP accumulator."""
    def __init__(self, stepper, phi, rng, steps, active, capture):
        self.phi, self.rng, self.stepper = phi.clone(), rng, stepper
        self.ep = torch.zeros(phi.shape[0], device=phi.device, dtype=phi.dtype)
        self.bad = torch.zeros((), device=phi.device, dtype=torch.bool)
        self.steps, self.active = steps, active
        self.graph = None
        if capture:
            state = rng.get_state()
            stream = torch.cuda.Stream(device=phi.device)
            stream.wait_stream(torch.cuda.current_stream(phi.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.eager()
            torch.cuda.current_stream(phi.device).wait_stream(stream)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(rng)
            with torch.cuda.graph(graph, stream=stream):
                self.eager()
            torch.cuda.current_stream(phi.device).wait_stream(stream)
            self.graph = graph
            rng.set_state(state)
            self.phi.copy_(phi)
            self.ep.zero_()
            self.bad.zero_()

    def eager(self):
        for _ in range(self.steps):
            normal = torch.randn((len(self.phi), 2, *self.phi.shape[-2:]),
                                 device=self.phi.device, dtype=self.phi.dtype, generator=self.rng)
            new = self.stepper(self.phi, normal)
            if self.active:
                mu = self.stepper.effective_active((self.phi + new) / 2)
                self.ep.add_((-mu * (new-self.phi)).sum((-2, -1))
                             * self.stepper.dx**2 / self.stepper.D)
            self.bad.logical_or_(~torch.isfinite(new).all() | (new.abs().max() > 30))
            self.phi.copy_(new)

    def run(self):
        if self.graph is None:
            self.eager()
        else:
            self.graph.replay()


def cluster_area(field, threshold, dx):
    """Number-average area of positive-phase 4-connected periodic components."""
    from scipy.ndimage import label
    labels, count = label(field > threshold)
    if not count:
        return 0.
    parent = np.arange(count+1)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for first, last in ((labels[0], labels[-1]), (labels[:, 0], labels[:, -1])):
        for a, b in zip(first, last):
            if a and b:
                parent[find(a)] = find(b)
    components = len({find(i) for i in range(1, count+1)})
    return np.count_nonzero(labels) * dx**2 / components


@torch.inference_mode()
def simulate(args):
    if args.dt <= 0 or not math.isfinite(args.dt):
        raise ValueError('dt must be finite and positive')
    burn = step_count(args.burn_time, args.dt)
    production = step_count(args.production_time, args.dt)
    if production < 1 or min(args.trajectories, args.diagnostic_every, args.movie_every,
                              args.cpu_threads, args.fps) < 1 or args.graph_steps < 0:
        raise ValueError('Invalid step, trajectory, thread, or FPS count')
    if not args.g or len(set(args.g)) != len(args.g) or not np.isfinite(args.g).all():
        raise ValueError('g values must be distinct and finite')
    if not math.isfinite(args.cluster_threshold) or args.seed < 0:
        raise ValueError('Invalid threshold or seed')
    if (args.output/'config.json').exists():
        raise FileExistsError('Output already contains an experiment; use --plot-only or a new --output')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    torch.set_num_threads(args.cpu_threads)
    root, lramb = _lramb_module(args.lramb_root)
    from lramb.torch_stencils import from_model
    params = lramb.Parameters(**{k: getattr(args, k) for k in
        ('n', 'dx', 'dt', 'a', 'b', 'kappa', 'lam', 'zeta', 'mobility', 'D', 'mean')}, seed=args.seed)
    model = lramb.ActiveModelB(params, lramb.KernelSpec('poisson', strength=1., ell=args.ell))
    stepper = from_model(model, device, torch.float64)
    batch = len(args.g)*args.trajectories
    # Absorb each coupling into its kernel; preserve the established solver.
    couplings = torch.tensor(args.g, device=device, dtype=torch.float64).repeat_interleave(args.trajectories)
    stepper.kernel = stepper.kernel[None] * couplings[:, None, None]
    fields = []
    for i in range(batch):
        model.rng = np.random.default_rng(args.seed+i)
        fields.append(model.initial_field(args.initial))
    phi = torch.tensor(np.stack(fields), device=device, dtype=torch.float64)
    initial_mass = phi.mean((-2, -1)).clone()
    rng = torch.Generator(device=device).manual_seed(args.seed)
    capture = device.type == 'cuda' and args.graph_steps > 0
    chunk_steps = math.gcd(args.graph_steps, math.gcd(args.diagnostic_every, args.movie_every)) if capture else 1
    # Phase boundaries may need a smaller divisor, so no updates are skipped.
    chunk_steps = math.gcd(chunk_steps, math.gcd(burn, production))
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), lramb_root=str(root), burn_steps=burn,
                  production_steps=production, effective_graph_steps=chunk_steps if capture else 0,
                  ep_definition='-sum(mu_active_eff(midpoint)*dphi)*dx^2/D; every solver step',
                  cluster_definition='number-average phi>threshold component area; periodic 4-connectivity',
                  complete=False)
    (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    frame_steps = sorted(set([0, production, *range(args.movie_every, production, args.movie_every)]))
    movie = np.lib.format.open_memmap(args.output/'fields.npy', mode='w+', dtype=np.float32,
                                     shape=(batch, len(frame_steps), args.n, args.n))
    trace_t, trace_var, ep_t, ep_trace = [], [], [], []
    def diagnostic(step, field):
        trace_t.append(step*args.dt)
        trace_var.append(field.var((-2, -1), correction=0).cpu().numpy())
    diagnostic(0, phi)
    started = time.perf_counter()
    for active, steps in ((False, burn), (True, production)):
        runner = Chunk(stepper, phi, rng, chunk_steps, active, capture)
        if active:
            movie[:, 0] = phi.cpu().numpy()
            frame = 1
            ep_t.append(0.)
            ep_trace.append(np.zeros(batch))
        next_report = time.perf_counter()+30
        for done in range(chunk_steps, steps+1, chunk_steps):
            runner.run()
            if done % args.diagnostic_every == 0 or done == steps:
                if bool(runner.bad):
                    raise FloatingPointError('Unstable field; reduce dt. Experiment is incomplete.')
                diagnostic((burn if active else 0)+done, runner.phi)
                if active:
                    ep_t.append(done*args.dt)
                    ep_trace.append(runner.ep.cpu().numpy().copy())
            if active and frame < len(frame_steps) and done == frame_steps[frame]:
                movie[:, frame] = runner.phi.cpu().numpy()
                frame += 1
            if time.perf_counter() >= next_report:
                print(f'{"production" if active else "burn"}: {done:,}/{steps:,} steps', flush=True)
                next_report = time.perf_counter()+30
        phi = runner.phi.clone()
    movie.flush()
    np.savez(args.output/'observables.npz', time=np.array(trace_t), variance=np.stack(trace_var),
             ep_time=np.array(ep_t), cumulative_ep=np.stack(ep_trace),
             movie_time=burn*args.dt+np.array(frame_steps)*args.dt,
             mass_error=(phi.mean((-2, -1))-initial_mass).cpu().numpy())
    config.update(complete=True, elapsed_seconds=time.perf_counter()-started)
    (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Simulation saved: {args.output}', flush=True)


def render(directory, movie_format, fps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    cfg = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    if not cfg['complete']:
        raise ValueError('Simulation is incomplete')
    data = np.load(directory/'observables.npz')
    fields = np.load(directory/'fields.npy', mmap_mode='r')
    count = cfg['trajectories']
    rates = data['cumulative_ep'][-1].reshape(-1, count)/cfg['production_time']
    areas = np.empty((len(cfg['g']), count))
    for gi, g in enumerate(cfg['g']):
        for rep in range(count):
            member = gi*count+rep
            target = directory/f'g_{gi:03d}_{g:g}'/f'traj_{rep:03d}'
            target.mkdir(parents=True, exist_ok=True)
            area = np.array([cluster_area(f, cfg['cluster_threshold'], cfg['dx']) for f in fields[member]])
            # Time average by trapezoidal quadrature, including a partial final interval.
            times = data['movie_time']
            areas[gi, rep] = np.sum((area[:-1]+area[1:])/2*np.diff(times))/cfg['production_time']
            np.savetxt(target/'cluster_area.csv', np.column_stack((times, area)),
                       delimiter=',', header='time,mean_cluster_area', comments='')
            fig, ax = plt.subplots()
            ax.plot(data['time'], data['variance'][:, member])
            ax.axvline(cfg['burn_time'], color='k', ls='--', label='burn-in ends')
            ax.set(xlabel='time', ylabel='spatial field variance', title=f'g={g:g}, trajectory {rep}')
            ax.legend(); fig.tight_layout(); fig.savefig(target/'field_fluctuation.png', dpi=150); plt.close(fig)
            fig, ax = plt.subplots()
            limit = max(1e-9, float(np.max(np.abs(fields[member]))))
            im = ax.imshow(fields[member, 0], cmap='RdBu_r', vmin=-limit, vmax=limit,
                           extent=(0, cfg['n']*cfg['dx'], 0, cfg['n']*cfg['dx']))
            fig.colorbar(im, ax=ax, label='field')
            title = ax.set_title('')
            def update(i):
                im.set_data(fields[member, i])
                title.set_text(f'g={g:g}, traj={rep}, t={times[i]:g}')
                return im, title
            animation = FuncAnimation(fig, update, frames=len(times), blit=False)
            writer = PillowWriter(fps=fps) if movie_format == 'gif' else FFMpegWriter(fps=fps)
            animation.save(target/f'steady_state.{movie_format}', writer=writer)
            plt.close(fig)
    def summary(values, ylabel, name):
        means = values.mean(1)
        sem = values.std(1, ddof=1)/np.sqrt(count) if count > 1 else np.full(len(means), np.nan)
        fig, ax = plt.subplots()
        for j in range(count):
            ax.scatter(cfg['g'], values[:, j], alpha=.35, color='C0')
        order = np.argsort(cfg['g'])
        ax.errorbar(np.array(cfg['g'])[order], means[order],
                    yerr=sem[order] if count > 1 else None, fmt='o-', color='k', label='mean +/- trajectory SEM')
        ax.set(xlabel='g', ylabel=ylabel); ax.legend(); fig.tight_layout()
        fig.savefig(directory/f'{name}.png', dpi=180); plt.close(fig)
        np.savetxt(directory/f'{name}.csv', np.column_stack((cfg['g'], means, sem, values)),
                   delimiter=',', header='g,mean,sem,'+','.join(f'traj_{i}' for i in range(count)), comments='')
    summary(rates, 'Total active-only EPR (EP / time)', 'ensemble_epr_vs_g')
    summary(rates*cfg['production_time'], 'Total active-only EP (production window)', 'ensemble_ep_vs_g')
    summary(areas, 'Mean cluster area (length squared)', 'cluster_size_vs_g')
    print(f'Plots and movies saved: {directory}', flush=True)


if __name__ == '__main__':
    args = parse_args()
    if not args.plot_only:
        simulate(args)
    render(args.output, args.movie_format, args.fps)
