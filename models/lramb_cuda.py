"""Optional CUDA-graph burn-in for the Poisson dataset generator.

Only launch scheduling changes: each replay performs ``chunk_steps`` full
Euler-Maruyama updates, including a fresh canonical rFFT after every update.
The cache holds one graph per worker, reused across trajectory seeds and kernels.
"""

import torch


class _BurnGraph:
    def __init__(self, step_fn, phi, ph, linear, q2, noise_scale,
                 dt_mobility, b, lam, dx, chunk_steps):
        self.phi, self.ph = phi.clone(), ph.clone()
        self.linear, self.q2 = linear.clone(), q2.clone()
        self.noise_scale = noise_scale.clone()
        self.error = torch.zeros((), dtype=phi.dtype, device=phi.device)
        self.bad = torch.zeros((), dtype=torch.bool, device=phi.device)
        self.rng = torch.Generator(device=phi.device).manual_seed(0)
        self.chunk_steps = chunk_steps

        def chunk():
            for _ in range(chunk_steps):
                normal = torch.randn(self.phi.shape, dtype=phi.dtype,
                                     device=phi.device, generator=self.rng)
                new, propagated = step_fn(
                    self.phi, self.ph, normal, self.linear, self.q2,
                    self.noise_scale, dt_mobility, b, lam, dx)
                canonical = torch.fft.rfft2(new, norm="ortho")
                self.error.copy_(torch.maximum(
                    self.error, (propagated - canonical).abs().max()))
                self.bad.logical_or_(~torch.isfinite(new).all() | (new.abs().max() > 30))
                # Stable input addresses are essential for subsequent replays.
                self.phi.copy_(new)
                self.ph.copy_(canonical)

        # Warm up JIT, FFT plans and the allocator on the capture side stream.
        stream = torch.cuda.Stream(device=phi.device)
        stream.wait_stream(torch.cuda.current_stream(phi.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.phi.copy_(phi)
                self.ph.copy_(ph)
                chunk()
        torch.cuda.current_stream(phi.device).wait_stream(stream)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        if not hasattr(self.graph, "register_generator_state"):
            raise RuntimeError(
                "This PyTorch version lacks CUDA graph custom-generator support; "
                "use --cuda-graph-steps 0 in a separate output directory")
        # Register the custom generator so replay advances noise rather than
        # repeating the captured random realization.
        self.graph.register_generator_state(self.rng)
        with torch.cuda.graph(self.graph, stream=stream):
            chunk()
        torch.cuda.current_stream(phi.device).wait_stream(stream)

    def run(self, phi, ph, linear, q2, noise_scale, rng, chunks):
        # All state manipulation is OUTSIDE capture. Warmup/capture must not
        # consume the trajectory's random sequence or change its initial field.
        self.phi.copy_(phi)
        self.ph.copy_(ph)
        self.linear.copy_(linear)
        self.q2.copy_(q2)
        self.noise_scale.copy_(noise_scale)
        self.error.zero_()
        self.bad.zero_()
        self.rng.set_state(rng.get_state())
        for _ in range(chunks):
            self.graph.replay()
            if bool(self.bad):
                raise FloatingPointError("Unstable CUDA graph burn-in trajectory; reduce dt")
        rng.set_state(self.rng.get_state())
        # Do not expose storage that the next trajectory will overwrite.
        return self.phi.clone(), self.ph.clone(), self.error.clone()


_cached_key = None
_cached_graph = None


def graph_burn_in(step_fn, phi, ph, linear, q2, noise_scale, rng, *,
                  dt_mobility, b, lam, dx, steps, chunk_steps):
    """Run complete chunks, returning (field, spectrum, max error, steps done).

    The caller executes any remaining steps eagerly with the advanced RNG state.
    Capture failures propagate; do not silently resume after a possibly invalid
    CUDA capture. The graph option is explicit and disabled by default.
    """
    global _cached_key, _cached_graph
    if phi.device.type != "cuda":
        raise ValueError("CUDA graph burn-in requires a CUDA device")
    chunks = steps // chunk_steps
    if not chunks:
        return phi, ph, torch.zeros((), dtype=phi.dtype, device=phi.device), 0
    key = (str(phi.device), tuple(phi.shape), phi.dtype, dt_mobility, b, lam,
           dx, chunk_steps, id(step_fn))
    if key != _cached_key:
        _cached_graph = None
        _cached_key = None
        with torch.cuda.device(phi.device):
            graph = _BurnGraph(step_fn, phi, ph, linear, q2, noise_scale,
                               dt_mobility, b, lam, dx, chunk_steps)
        _cached_graph, _cached_key = graph, key
    with torch.cuda.device(phi.device):
        field, spectrum, error = _cached_graph.run(
            phi, ph, linear, q2, noise_scale, rng, chunks)
    return field, spectrum, error, chunks * chunk_steps
