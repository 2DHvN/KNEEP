"""Focused g reports. Reference = spatially integrated active-only EP.

Fluctuation = within-trajectory temporal variance of saved-interval EPR,
then averaged over trajectories and fit seeds. Pixel normalization divides
variance by N**2; total normalization divides by mean predicted total EPR**2.
"""
from pathlib import Path
import csv
import json
import numpy as np


def statistics(record):
    dt = float(record['interval'])
    rate = np.asarray(record['predicted_increment']) / dt  # fit, trajectory, time, branch
    true = np.asarray(record['reference_increment'])[..., 2] / dt
    counts = np.asarray(record['branch_input_pixel_counts'])
    if counts.shape != (rate.shape[-1],) or np.any(counts <= 0):
        raise ValueError('Invalid branch input-pixel counts')
    # Equal weighting; never treat correlated times or shared-test fits as independent samples.
    pred_by_traj = rate.sum(-1).mean((0, 2))
    pred_by_fit = rate.sum(-1).mean((1, 2))
    true_by_traj = true.mean(1)
    def sem(x):
        return float(np.std(x, ddof=1)/np.sqrt(len(x))) if len(x) > 1 else float('nan')
    pred_mean, true_mean = float(pred_by_traj.mean()), float(true_by_traj.mean())
    variance = rate.var(2).mean((0, 1))
    return dict(pred=pred_mean, true=true_mean, pred_trajectory_sem=sem(pred_by_traj),
                true_trajectory_sem=sem(true_by_traj), pred_fit_sem=sem(pred_by_fit),
                ratio=pred_mean/true_mean if abs(true_mean) > 1e-10 else float('nan'),
                branch_mean=rate.mean((0, 1, 2)), variance=variance,
                per_pixel_variance=variance/counts**2,
                relative_variance=variance/pred_mean**2 if abs(pred_mean) > 1e-10
                    else np.full_like(variance, np.nan),
                relative_per_pixel_variance=variance/counts**2/pred_mean**2 if abs(pred_mean) > 1e-10
                    else np.full_like(variance, np.nan), counts=counts)


def save_g_figures(records, figure_dir, *, show=False, include_medium=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.poisson_range_plots import plot_training_history
    folder = Path(figure_dir)
    folder.mkdir(parents=True, exist_ok=True)
    def save(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches='tight')
        if not show:
            plt.close(fig)
    # Separate ell groups, so x=g comparisons never mix interaction ranges.
    groups = {}
    for record in records:
        ell = float(record['metadata']['condition'].get('solver_ell', record['ell']))
        groups.setdefault(ell, []).append(record)
    for ell, group in groups.items():
        group = sorted(group, key=lambda r: float(r['strength']))
        out = folder if len(groups) == 1 else folder/f'ell_{ell:g}'
        out.mkdir(parents=True, exist_ok=True)
        stats = [statistics(r) for r in group]
        for r, s in zip(group, stats):
            dest = out/r['condition_id']
            dest.mkdir(parents=True, exist_ok=True)
            save(plot_training_history([r]), dest/'loss.png')
            predicted = np.asarray(r['predicted_map']).sum(2).mean((0, 1))
            reference = np.asarray(r['reference_map']).mean(0)*float(r['dx'])**2
            maps = [predicted, reference[2]]
            titles = ['Predicted (fit + trajectory mean)', 'Active-only (trajectory mean)']
            common = max(np.abs(maps).max(), 1e-12)
            limits = [common, common]
            if include_medium:
                maps.append(reference.sum(0)); titles.append('Medium (separate color scale)')
                limits.append(max(np.abs(maps[-1]).max(), 1e-12))
            fig, axes = plt.subplots(1, len(maps), figsize=(5*len(maps), 4), squeeze=False)
            for ax, value, title, limit in zip(axes[0], maps, titles, limits):
                im = ax.imshow(value, origin='lower', cmap='RdBu_r', vmin=-limit, vmax=limit)
                ax.set(title=title, xlabel='x pixel', ylabel='y pixel')
                fig.colorbar(im, ax=ax, label='EP / time / site')
            fig.suptitle(f'g={r["strength"]:g}, ell={ell:g}; time-averaged maps')
            save(fig, dest/'ep_maps.png')
            radius = np.arange(len(s['branch_mean']))
            for key, label, name in (
                ('branch_mean', 'Mean branch EPR', 'branch_ep_average'),
                ('variance', 'Var_t(branch EPR)', 'branch_ep_fluctuation'),
                ('per_pixel_variance', 'Var_t(branch EPR) / N_pixels^2', 'branch_ep_fluctuation_per_pixel')):
                fig, ax = plt.subplots()
                ax.plot(radius, s[key], 'o-')
                ax.set(xlabel='Branch radius (pixels)', ylabel=label, title=f'g={r["strength"]:g}')
                if key != 'branch_mean':
                    ax.set_yscale('symlog', linthresh=max(float(np.nanmax(s[key]))*1e-5, 1e-15))
                save(fig, dest/(name+'.png'))
            np.savetxt(dest/'branch_statistics.csv', np.column_stack((radius, s['counts'],
                s['branch_mean'], s['variance'], s['per_pixel_variance'], s['relative_variance'])),
                delimiter=',', header='radius,input_pixels,mean_epr,variance,variance_per_pixel_squared,variance_per_total_epr_squared', comments='')
        for normalized in (False, True):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            keys = ('relative_variance', 'relative_per_pixel_variance') if normalized else ('variance', 'per_pixel_variance')
            for r, s in zip(group, stats):
                for ax, key in zip(axes, keys):
                    ax.plot(np.arange(len(s[key])), s[key], 'o-', label=f'g={r["strength"]:g}')
            for i, ax in enumerate(axes):
                ax.set(xlabel='Branch radius (pixels)',
                       ylabel='Var_t(branch EPR)' + (' / N_pixels^2' if i else '')
                           + (' / <total predicted EPR>^2' if normalized else ''))
                finite = np.concatenate([s[keys[i]][np.isfinite(s[keys[i]])] for s in stats])
                ax.set_yscale('symlog', linthresh=max(float(finite.max())*1e-5 if len(finite) else 0., 1e-15))
                ax.legend()
            save(fig, out/('branch_fluctuation_overlay_normalized.png' if normalized else 'branch_fluctuation_overlay.png'))
        g = [r['strength'] for r in group]
        fig, ax = plt.subplots()
        for key, label in (('true', 'Active-only reference'), ('pred', 'Prediction')):
            ax.errorbar(g, [s[key] for s in stats], yerr=[s[key+'_trajectory_sem'] for s in stats], fmt='o-', label=label)
        ax.set(xlabel='g', ylabel='Total EPR', title='Mean +/- test-trajectory SEM (fit-averaged prediction)')
        ax.legend(); save(fig, out/'epr_vs_g.png')
        fig, ax = plt.subplots()
        ax.plot(g, [s['ratio'] for s in stats], 'o-')
        ax.axhline(1, color='gray', ls='--')
        ax.set(xlabel='g', ylabel='Mean predicted / mean active-only EPR')
        save(fig, out/'pred_true_ratio_vs_g.png')
        columns = ['g', 'ell', 'true', 'pred', 'true_trajectory_sem', 'pred_trajectory_sem', 'pred_fit_sem', 'ratio']
        with (out/'epr_vs_g.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            for r, s in zip(group, stats):
                writer.writerow(dict(g=r['strength'], ell=ell, **{k:s[k] for k in columns[2:]}))
    (folder/'definitions.json').write_text(json.dumps(dict(
        reference='active-only, midpoint, every solver step; spatial sum',
        fluctuation='population temporal variance within each trajectory, then trajectory/fit mean',
        pixels='actual branch force-input pixels; variance divided by count squared',
        total_normalization='variance / squared mean total predicted EPR; near-zero denominator => NaN',
        uncertainty='test-trajectory SEM after averaging fits; fit SEM reported separately in CSV',
        ratio='ratio of ensemble means; abs(reference)<=1e-10 => NaN'), indent=2), encoding='utf-8')
    if show:
        plt.show()
