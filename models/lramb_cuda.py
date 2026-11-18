"""Optional CUDA-graph burn-in for the Poisson dataset generator.

Each replay performs ``chunk_steps`` real-space Euler-Maruyama updates.
The diagnostic rFFT is constructed once after burn-in. The cache holds one
graph per worker and supports a full batch of independent fields.
"""

import torch


class _BurnGraph:
    def __init__(self, stepper, phi, ph, chunk_steps):
        self.phi, self.ph = phi.clone(), ph.clone()
        self.stepper = stepper
        self.error = torch.zeros((), dtype=phi.dtype, device=phi.device)
        self.bad = torch.zeros((), dtype=torch.bool, device=phi.device)
        self.rng = torch.Generator(device=phi.device).manual_seed(0)
        self.chunk_steps = chunk_steps

        def chunk():
            for _ in range(chunk_steps):
                normal = torch.randn((*self.phi.shape[:-2],2,*self.phi.shape[-2:]), dtype=phi.dtype,
                                     device=phi.device, generator=self.rng)
                new = self.stepper(self.phi, normal)
                self.bad.logical_or_(~torch.isfinite(new).all() | (new.abs().max() > 30))
                # Stable input addresses are essential for subsequent replays.
                self.phi.copy_(new)

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

    def run(self, phi, ph, stepper, rng, chunks):
        # All state manipulation is OUTSIDE capture. Warmup/capture must not
        # consume the trajectory's random sequence or change its initial field.
        self.phi.copy_(phi)
        self.ph.copy_(ph)
        self.stepper.kernel.copy_(stepper.kernel)
        self.error.zero_()
        self.bad.zero_()
        self.rng.set_state(rng.get_state())
        for _ in range(chunks):
            self.graph.replay()
            if bool(self.bad):
                raise FloatingPointError("Unstable CUDA graph burn-in trajectory; reduce dt")
        rng.set_state(self.rng.get_state())
        # Do not expose storage that the next trajectory will overwrite.
        return self.phi.clone(), torch.fft.rfft2(self.phi, norm="ortho"), self.error.clone()


_cached_key = None
_cached_graph = None


def graph_burn_in(stepper, phi, ph, rng, *, steps, chunk_steps):
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
    key = (str(phi.device), tuple(phi.shape), phi.dtype, chunk_steps,
           stepper.a, stepper.b, stepper.kappa, stepper.lam, stepper.zeta,
           stepper.dx, stepper.dt, stepper.mobility, stepper.D, stepper.strength)
    if key != _cached_key:
        _cached_graph = None
        _cached_key = None
        with torch.cuda.device(phi.device):
            graph = _BurnGraph(stepper, phi, ph, chunk_steps)
        _cached_graph, _cached_key = graph, key
    with torch.cuda.device(phi.device):
        field, spectrum, error = _cached_graph.run(
            phi, ph, stepper, rng, chunks)
    return field, spectrum, error, chunks * chunk_steps
