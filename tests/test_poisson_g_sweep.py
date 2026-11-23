"""Small numerical checks; no CUDA or long experiment required."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lramb_poisson import (dataset_preset, dataset_conditions, initial_field,
    _lramb_module, _condition_model, _shell_multipliers, _run_numpy, _run_torch)
from utils.poisson_g_plots import statistics
from utils.training import TrainingConfig, train_model


class GSweepTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(55)
        train = torch.randn(2, 5, 1, 8, 8, generator=rng)
        val = torch.randn(2, 4, 1, 8, 8, generator=rng)
        cfg = TrainingConfig(iterations=8, train_batch_size=4, validation_batch_size=4,
            validate_every=2, lr_scheduler='plateau', lr_plateau_patience=0,
            lr_plateau_threshold=10.)
        options = dict(model_seed=17, device='cpu', n_components=1,
                       hidden_channels=2, hidden_layers=1, max_distance=1, progress=False)
        full = train_model(train, val, cfg, **options)
        first = train_model(train, val, replace(cfg, iterations=4), **options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.pt'
            torch.save(first.resume_state, path)
            saved = torch.load(path, weights_only=True)
        resumed = train_model(train, val, cfg, resume_state=saved, **options)
        self.assertEqual(full.history_iterations, resumed.history_iterations)
        self.assertEqual(full.learning_rates, resumed.learning_rates)
        self.assertEqual(full.validation_losses, resumed.validation_losses)
        self.assertEqual(full.resume_state['scheduler_state'], resumed.resume_state['scheduler_state'])
        for key in full.resume_state['model_state']:
            torch.testing.assert_close(full.resume_state['model_state'][key],
                                       resumed.resume_state['model_state'][key], rtol=0, atol=0)
        torch.testing.assert_close(full.resume_state['sampler_state'], resumed.resume_state['sampler_state'])

    def test_cosine_extension_does_not_restart_learning_rate(self):
        torch.set_num_threads(1)
        video = torch.randn(1, 3, 1, 8, 8, generator=torch.Generator().manual_seed(5))
        cfg = TrainingConfig(iterations=2, train_batch_size=2, validation_batch_size=2,
                             validate_every=1, lr_scheduler='cosine', lr_min=1e-5)
        options = dict(model_seed=17, device='cpu', n_components=1,
                       hidden_channels=2, hidden_layers=1, max_distance=1, progress=False)
        first = train_model(video, video, cfg, **options)
        resumed = train_model(video, video, replace(cfg, iterations=4),
                              resume_state=first.resume_state, **options)
        self.assertEqual(resumed.learning_rates[-2:], (cfg.lr_min, cfg.lr_min))

    def test_signed_conditions_and_initial_mass(self):
        cfg = dataset_preset('g_steady')
        self.assertEqual(cfg.dt*cfg.burn_steps, 20000)
        conditions = dataset_conditions(cfg)
        self.assertEqual(len(conditions), 7)
        self.assertEqual({c['ell'] for c in conditions}, {2.})
        self.assertEqual(sum(c['strength'] == 0 for c in conditions), 1)
        self.assertEqual(len({c['condition_id'] for c in conditions}), 7)
        _, module = _lramb_module(None)
        fields = [initial_field(_condition_model(module, cfg, c, 17), cfg) for c in conditions]
        for field in fields:
            np.testing.assert_array_equal(field, fields[0])
        self.assertAlmostEqual(fields[0].mean(), -.5972084164192639)

    def test_active_movies_integrate_to_full_step_reference(self):
        torch.set_num_threads(1)
        cfg = replace(dataset_preset('quick'), ells=(2.,), strengths=(-.02, 0., .02),
                      movie_window_steps=7)  # includes a partial final window
        _, module = _lramb_module(None)
        for condition in dataset_conditions(cfg):
            for backend in ('numpy', 'torch'):
                model = _condition_model(module, cfg, condition, 17)
                local = replace(cfg, strength=condition['strength'])
                kwargs = dict(capture_movie=True)
                result = (_run_numpy(model, local, _shell_multipliers(model), **kwargs)
                          if backend == 'numpy' else
                          _run_torch(model, local, _shell_multipliers(model), 'cpu', **kwargs))
                movie = result['movie']
                reference = np.asarray(result['reference_increment'])
                for i, (start, end) in enumerate(zip(movie['window_start'], movie['window_end'])):
                    np.testing.assert_allclose(movie['active_rate_maps'][i].sum()*movie['window_duration'][i],
                                               reference[start:end, 2].sum(), rtol=1e-10, atol=1e-11)
                np.testing.assert_allclose(np.average(movie['active_rate_maps'], axis=0,
                    weights=movie['window_duration']), np.asarray(result['reference_map'])[2]*cfg.dx**2,
                    rtol=1e-10, atol=1e-11)

    def test_statistics_preserve_within_trajectory_variance(self):
        # Different trajectory offsets must not inflate temporal fluctuations.
        pred = np.array([[[[1., 2.], [3., 4.]], [[11., 12.], [13., 14.]]]])
        reference = np.zeros((2, 2, 3)); reference[..., 2] = 2.
        result = statistics(dict(predicted_increment=pred, reference_increment=reference,
                                 interval=.5, branch_input_pixel_counts=[1, 2]))
        np.testing.assert_allclose(result['variance'], [4, 4])
        np.testing.assert_allclose(result['per_pixel_variance'], [4, 1])
        self.assertEqual(result['pred'], 30.)
        self.assertEqual(result['true'], 4.)
        self.assertEqual(result['ratio'], 7.5)
        np.testing.assert_allclose(result['relative_variance'], np.array([4, 4])/900)

    def test_zero_denominators(self):
        result = statistics(dict(predicted_increment=np.zeros((1, 1, 2, 1)),
            reference_increment=np.zeros((1, 2, 3)), interval=1., branch_input_pixel_counts=[1]))
        self.assertTrue(np.isnan(result['ratio']))
        self.assertTrue(np.isnan(result['relative_variance']).all())


if __name__ == '__main__':
    unittest.main()
