import os
import pathlib
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.transforms import ScaledTranslation, IdentityTransform

import util

SOLARIZED = dict(
    base03='#002b36',
    base02='#073642',
    base01='#586e75',
    base00='#657b83',
    base0='#839496',
    base1='#93a1a1',
    base2='#eee8d5',
    base3='#fdf6e3',
    yellow='#b58900',
    orange='#cb4b16',
    red='#dc322f',
    magenta='#d33682',
    violet='#6c71c4',
    blue='#268bd2',
    cyan='#2aa198',
    green='#859900')
# SOLARIZED_CMAP = LinearSegmentedColormap.from_list(
#     'solarized', (SOLARIZED['base03'], SOLARIZED['base1'], SOLARIZED['base3']))
_solarized_values = np.array([15, 20, 45, 50, 60, 65, 92, 97])
_normalised_solarized_values = (_solarized_values - 15) / 82
SOLARIZED_CMAP = LinearSegmentedColormap.from_list(
    'solarized', ((_normalised_solarized_values[0], SOLARIZED['base03']),
                  (_normalised_solarized_values[1], SOLARIZED['base02']),
                  (_normalised_solarized_values[2], SOLARIZED['base01']),
                  (_normalised_solarized_values[3], SOLARIZED['base00']),
                  (_normalised_solarized_values[4], SOLARIZED['base0']),
                  (_normalised_solarized_values[5], SOLARIZED['base1']),
                  (_normalised_solarized_values[6], SOLARIZED['base2']),
                  (_normalised_solarized_values[7], SOLARIZED['base3'])))
SOLARIZED_CMAP.set_under(SOLARIZED['base03'])
SYMMETRIC_CMAP = LinearSegmentedColormap.from_list(
    'solarized_symmetric', (SOLARIZED['blue'], SOLARIZED['base1'], SOLARIZED['orange']))
LINEAR_CMAP = LinearSegmentedColormap.from_list(
    'solarized_symmetric', (SOLARIZED['violet'],
                            SOLARIZED['blue'],
                            SOLARIZED['cyan'],
                            SOLARIZED['green'],
                            SOLARIZED['yellow'],
                            SOLARIZED['orange'],
                            SOLARIZED['red'],
                            SOLARIZED['magenta']))

KEY_TO_LABEL = dict(
    GD='GD Minimal',
    GDmwd='GD Full',
    SGD='SGD Minimal',
    SGDmwd='SGD Full',
    Adam='Adam',
    KFAC='K-FAC',
    SGDQLR_Undamped_Hessian='',
    SGDQLR_Damped_Hessian='',
    AdamQLR_Undamped_Hessian='AdamQLR Undamped (Hessian Curvature)',
    AdamQLR_Damped_Hessian='AdamQLR (Tuned, Hessian)',
    AdamQLR_Damped_Hessian_DecreasingLossDamping='AdamQLR',
    AdamQLR_Damped_Hessian_NoHPO_SFN='AdamQLR (Untuned)',
    AdamQLR_Damped_AdamDampedCurvature='AdamQLR',
    AdamQLR_NoHPO='AdamQLR (Untuned)',
    AdamQLR_NoHPO_Unclipped='AdamQLR (Untuned, Unclipped)',
    AdamQLR_NoHPO_DecreasingLossDamping='AdamQLR (Untuned)',
    SGDQLR_Undamped='',
    SGDQLR_Damped='',
    AdamQLR_Undamped='AdamQLR (Undamped)',
    AdamQLR_Undamped_Clipped='AdamQLR Undamped (Fisher Curvature, Clipped)',
    AdamQLR_Damped='AdamQLR (Tuned)',
    AdamQLR_Damped_Clipped='AdamQLR Damped (Fisher Curvature, Clipped)',
    AdamQLR_Damped_Enveloped='AdamQLR Damped (Fisher Curvature, Enveloped)',

    SGDQLR_Undamped_Fisher='',
    SGDQLR_Damped_Fisher='',
    AdamQLR_Undamped_Fisher='AdamQLR Undamped (Fisher Curvature)',
    AdamQLR_Damped_Fisher='AdamQLR (Tuned, Fisher)',
)

KEY_TO_STYLE = dict(
    GD=dict(color=SOLARIZED['violet']),
    GDmwd=dict(color=SOLARIZED['blue']),
    SGD=dict(color=SOLARIZED['violet']),
    SGDmwd=dict(color=SOLARIZED['blue']),
    Adam=dict(color=SOLARIZED['green']),
    KFAC=dict(color=SOLARIZED['orange']),
    # SGDQLR_Undamped_Hessian=dict(color=SOLARIZED['orange'], linestyle='dashed'),
    # SGDQLR_Damped_Hessian=dict(color=SOLARIZED['orange']),
    AdamQLR_Undamped_Hessian=dict(color=SOLARIZED['yellow']),
    AdamQLR_Damped_Hessian=dict(color=SOLARIZED['red']),
    AdamQLR_Damped_Hessian_DecreasingLossDamping=dict(color=SOLARIZED['magenta']),
    AdamQLR_Damped_Hessian_NoHPO_SFN=dict(color=SOLARIZED['base03']),
    AdamQLR_Damped_AdamDampedCurvature=dict(color=SOLARIZED['magenta']),
    AdamQLR_NoHPO=dict(color=SOLARIZED['base03']),
    AdamQLR_NoHPO_Unclipped=dict(color=SOLARIZED['base03']),
    AdamQLR_NoHPO_DecreasingLossDamping=dict(color=SOLARIZED['base03']),
    # SGDQLR_Undamped=dict(color=SOLARIZED['orange'], linestyle='dotted'),
    # SGDQLR_Damped=dict(color=SOLARIZED['orange'], linestyle='dashdot'),
    AdamQLR_Undamped=dict(color=SOLARIZED['yellow']),
    AdamQLR_Undamped_Clipped=dict(color=SOLARIZED['orange'], linestyle='dashed'),
    AdamQLR_Damped=dict(color=SOLARIZED['magenta']),
    AdamQLR_Damped_Clipped=dict(color=SOLARIZED['magenta']),
    AdamQLR_Damped_Enveloped=dict(color=SOLARIZED['magenta']),

    # SGDQLR_Undamped_Fisher=dict(color=SOLARIZED['orange'], linestyle='dotted'),
    # SGDQLR_Damped_Fisher=dict(color=SOLARIZED['orange'], linestyle='dashdot'),
    AdamQLR_Undamped_Fisher=dict(color=SOLARIZED['magenta'], linestyle='dotted'),
    AdamQLR_Damped_Fisher=dict(color=SOLARIZED['magenta']),
)


def key_to_sort_order(x):
    algorithm_name = x.split(' ')[-1]
    if algorithm_name.startswith('X__'): return 999
    return tuple(KEY_TO_LABEL.keys()).index(algorithm_name)


@contextmanager
def inhibit_plt_show():
    original_show = plt.show
    plt.show = lambda: None
    try:
        yield
    finally:
        plt.show = original_show


def save_plot(func):
    def wrapper_function(*args, **kwargs):
        root_directory = kwargs.get('root_directory', args[0])
        root_name = pathlib.Path(root_directory).name
        plot_name = ' '.join(
            (root_name,
             kwargs.get('index', ''),
             (('Log_' if kwargs.get('log_x_axis', False) else '')
              + kwargs.get('metric', '/').replace('/', '_'))))
        def save():
            plt.gcf().set_size_inches(20, 11.25)
            plt.savefig(f'./plots/{plot_name}.pdf')
            plt.close()

        with inhibit_plt_show():
            return_value = func(*args, **kwargs)
            save()

        return return_value

    return wrapper_function


def reconstruct_plots(*directories, **kwargs):
    for directory in directories:
        for log_x_axis in (False, True):
            for metric in ('Loss/Training', 'Loss/Test'):
                save_plot(plot_best_run_envelopes)(directory,
                                                   metric=metric,
                                                   log_x_axis=log_x_axis,
                                                   **kwargs)


def get_pivoted_metric_evolution(data, num_sig_figs, index='wall_time'):
    def pivot_aggregator(x):
        # Use the latest loss recorded at each time step,
        # falling back to the final element if this isn't well-defined,
        # with an explicit isfinite to dodge infinities
        if any(np.isfinite(x)):
            x = x[np.isfinite(x)]
            return x[x.last_valid_index()]
        else:
            return x.iloc[-1]

    if index == 'wall_time':
        relative_times = pd.to_timedelta(
            data
            .groupby('dir_name', group_keys=False)
            ['wall_time']
            .apply(lambda group: util.round_sig_figs(
                group - group.min(), num_sig_figs)),
            unit='s')

        data_series = (data
                        .assign(wall_time=relative_times)
                        .pivot_table(
                            index='wall_time',
                            columns='dir_name',
                            values='value',
                            aggfunc=pivot_aggregator)
                        .interpolate(
                            method='time'))
        return data_series.set_index(
            data_series.index.total_seconds())
    elif index == 'step':
        data_series = (data
                        .pivot_table(
                            index='step',
                            columns='dir_name',
                            values='value',
                            aggfunc=pivot_aggregator))
        return data_series


def bootstrap_aggregate(data, aggregation, num_bootstrapped_datasets):
    def _aggregation(x, **kwargs):
        return getattr(x, aggregation)(**kwargs)

    aggregated_samples = pd.concat(
        (_aggregation(
            data.sample(frac=1, replace=True, axis=1), axis=1)
         for _ in range(num_bootstrapped_datasets)),
        axis='columns')
    mean_statistic = aggregated_samples.agg('mean', axis='columns')
    std_statistic = aggregated_samples.agg('std', axis='columns')
    return mean_statistic, std_statistic


def plot_best_run_envelopes(root_directory,
                            metric='Loss/Test',
                            aggregation='mean',
                            num_sig_figs=3,
                            num_bootstrapped_datasets=50,
                            log_x_axis=False,
                            index='wall_time',
                            included_algorithms=(),
                            remove_divergences=False,
                            break_x_axis=False):
    fig, axes = plt.subplots(1, 2 if break_x_axis else 1,
                             sharey=True,
                             width_ratios=(9, 1) if break_x_axis else None)
    fig.subplots_adjust(wspace=0.05)
    if not isinstance(axes, np.ndarray):
        axes = axes,
    max_time = float('-inf')
    for algorithm_directory in sorted(os.scandir(root_directory),
                                      key=lambda x: key_to_sort_order(x.name)):
        label = algorithm_directory.name
        if label.startswith('X__'):
            continue
        if included_algorithms and label not in included_algorithms:
            continue
        data = util.pandas_from_tensorboard(algorithm_directory)
        filtered_data = data[data['tag'] == metric]
        pivoted_data = get_pivoted_metric_evolution(filtered_data, num_sig_figs, index=index)
        if remove_divergences:
            undiverged_runs = pivoted_data.iloc[-1] < pivoted_data.iloc[0]
            pivoted_data = pivoted_data.loc[:, undiverged_runs]
        mean_evolution, std_evolution = bootstrap_aggregate(pivoted_data,
                                                            aggregation,
                                                            num_bootstrapped_datasets)
        for ax in axes:
            mean_evolution.plot(label=KEY_TO_LABEL[label],
                                **KEY_TO_STYLE[label],
                                ax=ax)
            ax.fill_between(std_evolution.index,
                            mean_evolution - std_evolution,
                            mean_evolution + std_evolution,
                            alpha=0.4,
                            color=KEY_TO_STYLE[label].get('color', None))
            max_time = max(max_time, mean_evolution.index[-1])

    if break_x_axis:
        axes[0].spines.right.set_visible(False)
        axes[0].yaxis.tick_left()
        axes[-1].spines.left.set_visible(False)
        axes[-1].yaxis.tick_right()
        axes[-1].tick_params(labelright=False)
        axes[-1].set_xlabel('')

        axes[0].set_xlim(0, break_x_axis)
        axes[-1].set_xlim(break_x_axis, None)

        axes[-1].set_xticks((break_x_axis, max_time))
        axes[-1].get_xticklabels()[0].set_visible(False)
        shared_label = axes[0].get_xticklabels()[-1]

        axis_spine = axes[0].spines.bottom
        for y_val in (0, 1):
            connection = ConnectionPatch(
                xyA=(0, y_val),
                xyB=(1, y_val),
                coordsA='axes fraction',
                coordsB='axes fraction',
                axesA=axes[1],
                axesB=axes[0],
                color=axis_spine.get_edgecolor(),
                linewidth=axis_spine.get_linewidth(),
                joinstyle=axis_spine.get_joinstyle())
            axes[1].add_artist(connection)

        # Matplotlib transforms act on points, not offsets,
        # so convert start and end points to display co-ordinates first.
        # Unclear what's responsible for the remaining position error
        translate_start, translate_end = connection.get_path().vertices[[0, 1]]
        display_offset = (axes[1].transData.transform(translate_start)
                          - axes[1].transData.transform(translate_end))*2/3
        shared_label.set_transform(
            shared_label.get_transform()
            + ScaledTranslation(*display_offset, IdentityTransform()))
        # Tick lines come in pairs, so [-1] is the invisible line in
        # the tick train at the top of the plot
        axes[0].get_xticklines()[-2].set_marker([(0, 0), (1, -1)])
        axes[0].get_xticklines()[-2].set_markersize(7)
        axes[-1].get_xticklines()[0].set_marker([(0, 0), (-1, -1)])
        axes[-1].get_xticklines()[0].set_markersize(7)
        axes[-1].get_xticklines()[0].set_fillstyle('none')

        # break_mark_kwargs = dict(
        #              marker=[(-0.5, -1), (0.5, 1)],
        #              markersize=12,
        #              linestyle='none',
        #              color=SOLARIZED['base1'],
        #              mec=SOLARIZED['base1'],
        #              mew=1,
        #              clip_on=False)
        # axes[0].plot([1, 1], [1, 0],
        #              transform=axes[0].transAxes,
        #              **break_mark_kwargs)
        # axes[1].plot([0, 0], [0, 1],
        #              transform=axes[1].transAxes,
        #              **break_mark_kwargs)
    legend_handles, _ = plt.gca().get_legend_handles_labels()
    legend_handles.append(
        Patch(facecolor='#aaa', alpha=0.4, label='± Standard Deviation'))
    if index == 'wall_time':
        axes[0].set_xlabel("Runtime (s)")
    elif index == 'step':
        axes[0].set_xlabel("Step")
    axes[0].set_ylabel(metric)
    if log_x_axis: plt.xscale('log')
    plt.yscale('log')
    axes[0].legend(handles=legend_handles)
    plt.show()
    return axes


def plot_rosenbrock_paths(root_directory, included_algorithms=()):
    base_grid = np.meshgrid(np.linspace(-1.5, 1.5, 1000),
                            np.linspace(-1.8, 1.2, 1000))
    rosenbrock_values = util.rosenbrock(*base_grid, a=1, b=100)
    background = plt.contourf(*base_grid,
                              rosenbrock_values,
                              norm='log',
                              # levels=100,
                              levels=np.logspace(-4, 4, 81),
                              cmap=SOLARIZED_CMAP,
                              extend='min')

    data = util.pandas_from_tensorboard(root_directory)
    x_trajectories = data[
        data['tag'] == 'Position/x'
    ].pivot_table(
            index='step',
            columns='dir_name',
            values='value')
    y_trajectories = data[
        data['tag'] == 'Position/y'
    ].pivot_table(
            index='step',
            columns='dir_name',
            values='value')
    for key in sorted(x_trajectories, key=key_to_sort_order):
        label = key.split(' ')[1]
        if included_algorithms and label not in included_algorithms:
            continue
        plt.plot([1, *x_trajectories[key]],
                 [-1, *y_trajectories[key]],
                 '-o',
                 label=KEY_TO_LABEL[label],
                 **KEY_TO_STYLE[label])

    # plt.scatter(1, -1, marker="x", color='k', label='Initial Point', zorder=10)
    plt.scatter(1, 1, marker="*", color='k', label='Minimum', zorder=10)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.8, 1.2)
    plt.gca().set_aspect('equal')
    plt.gcf().colorbar(
        background,
        ticks=np.logspace(-4, 4, 9),
        label='Rosenbrock Function')
    # plt.colorbar(cmap=SOLARIZED_CMAP, label='Log Rosenbrock')
    plt.legend()
    plt.show()


def plot_ablation_trends(root_directory,
                         metric='Loss/Test',
                         aggregation='mean',
                         num_sig_figs=3,
                         num_bootstrapped_datasets=50,
                         log_x_axis=False,
                         included_algorithms=(),
                         break_x_axis=False):
    fig, axes = plt.subplots(1, 2 if break_x_axis else 1,
                             sharey=True,
                             width_ratios=(9, 1) if break_x_axis else None)
    fig.subplots_adjust(wspace=0.05)
    if not isinstance(axes, np.ndarray):
        axes = axes,
    max_time = float('-inf')
    for algorithm_directory in sorted(os.scandir(root_directory),
                                      key=lambda x: float(x.name.split('_')[-1])):
        label = algorithm_directory.name
        if label.startswith('X__'):
            continue
        if included_algorithms and label not in included_algorithms:
            continue
        data = util.pandas_from_tensorboard(algorithm_directory)
        filtered_data = data[data['tag'] == metric]
        pivoted_data = get_pivoted_metric_evolution(filtered_data, num_sig_figs)
        mean_evolution, std_evolution = bootstrap_aggregate(pivoted_data,
                                                            aggregation,
                                                            num_bootstrapped_datasets)
        split_label = label.split('_')
        ablation_variable = split_label[0]
        label_value = float(split_label[-1])
        match ablation_variable:
            case 'Amplification':
                label_mantissa = np.log2(label_value)
                label_text = f'$k = 2^{{{label_mantissa.round(decimals=1)}}}$'
                colour = SYMMETRIC_CMAP((label_mantissa + 1) / 2)
            case 'BatchSize':
                label_text = int(label_value)
                colour = LINEAR_CMAP(np.log2(label_value / 50) / 6)
            case 'InitialDamping':
                label_mantissa = np.log10(label_value)
                label_text = f'$\\lambda_0 = 10^{{{label_mantissa.round(decimals=1)}}}$'
                colour = LINEAR_CMAP((np.log10(label_value) + 8) / 8)
            case 'LRClipping':
                label_mantissa = np.log10(label_value)
                label_text = f'$\\alpha_\\mathrm{{max}} = 10^{{{label_mantissa.round(decimals=1)}}}$'
                colour = LINEAR_CMAP((np.log10(label_value) + 4) / 5)
            case 'SteppingFactor':
                label_mantissa = np.log2(label_value)
                label_text = f'$\\omega_\\mathrm{{inc}} = \\frac{{1}}{{\\omega_\\mathrm{{dec}}}} = 2^{{{label_mantissa.round(decimals=1)}}}$'
                colour = LINEAR_CMAP(np.log2(label_value) / 2)
            case _:
                raise ValueError(f'Unknown prefix {ablation_variable}')
        for ax in axes:
            mean_evolution.plot(color=colour,
                                label=label_text,
                                ax=ax)
            ax.fill_between(std_evolution.index,
                            mean_evolution - std_evolution,
                            mean_evolution + std_evolution,
                            color=colour,
                            alpha=0.4)
            max_time = max(max_time, mean_evolution.index[-1])

    if break_x_axis:
        axes[0].spines.right.set_visible(False)
        axes[0].yaxis.tick_left()
        axes[-1].spines.left.set_visible(False)
        axes[-1].yaxis.tick_right()
        axes[-1].tick_params(labelright=False)
        axes[-1].set_xlabel('')

        axes[0].set_xlim(0, break_x_axis)
        axes[-1].set_xlim(break_x_axis, None)

        axes[-1].set_xticks((break_x_axis, max_time))
        axes[-1].get_xticklabels()[0].set_visible(False)
        shared_label = axes[0].get_xticklabels()[-1]

        axis_spine = axes[0].spines.bottom
        for y_val in (0, 1):
            connection = ConnectionPatch(
                xyA=(0, y_val),
                xyB=(1, y_val),
                coordsA='axes fraction',
                coordsB='axes fraction',
                axesA=axes[1],
                axesB=axes[0],
                color=axis_spine.get_edgecolor(),
                linewidth=axis_spine.get_linewidth(),
                joinstyle=axis_spine.get_joinstyle())
            axes[1].add_artist(connection)

        # Matplotlib transforms act on points, not offsets,
        # so convert start and end points to display co-ordinates first.
        # Unclear what's responsible for the remaining position error
        translate_start, translate_end = connection.get_path().vertices[[0, 1]]
        display_offset = (axes[1].transData.transform(translate_start)
                          - axes[1].transData.transform(translate_end))*2/3
        shared_label.set_transform(
            shared_label.get_transform()
            + ScaledTranslation(*display_offset, IdentityTransform()))
        # Tick lines come in pairs, so [-1] is the invisible line in
        # the tick train at the top of the plot
        axes[0].get_xticklines()[-2].set_marker([(0, 0), (1, -1)])
        axes[0].get_xticklines()[-2].set_markersize(7)
        axes[-1].get_xticklines()[0].set_marker([(0, 0), (-1, -1)])
        axes[-1].get_xticklines()[0].set_markersize(7)
        axes[-1].get_xticklines()[0].set_fillstyle('none')

    legend_handles, _ = plt.gca().get_legend_handles_labels()
    if len(legend_handles) > 10:
        legend_handles = [legend_handles[0],
                          legend_handles[1],
                          Patch(alpha=0.0, label=r'$\vdots$'),
                          legend_handles[len(legend_handles)//2],
                          Patch(alpha=0.0, label=r'$\vdots$'),
                          legend_handles[-2],
                          legend_handles[-1],
                          ]
    legend_handles.append(
        Patch(facecolor='#aaa', alpha=0.4, label='± Standard Deviation'))
    axes[0].set_xlabel("Runtime (s)")
    axes[0].set_ylabel(metric)
    if log_x_axis: plt.xscale('log')
    plt.yscale('log')
    axes[0].legend(handles=legend_handles)
    plt.show()
    return axes
