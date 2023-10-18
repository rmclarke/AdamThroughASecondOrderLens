import numpy as np
import pandas as pd
import pytest

import plot
import util


def get_sample_data():
    return util.pandas_from_tensorboard('/scratch/dir/ImprovingKFAC/ray/2023-02-27T11:47:22.506393__uci_energy__KFACwithDynamicKroneckerCorrections_Explicit_TrueApproxGrad__ASHA')

def possible_metrics():
    algorithm_data = get_sample_data()
    return algorithm_data['tag'].unique()


@pytest.fixture(params=possible_metrics())
def metric_data(request):
    algorithm_data = get_sample_data()
    return algorithm_data[algorithm_data['tag'] == request.param]


@pytest.fixture
def pivoted_data(metric_data):
    return plot.get_pivoted_metric_evolution(metric_data, 3)


def test_get_pivoted_metric_evolution(metric_data, pivoted_data):
    assert set(pivoted_data.columns) == set(metric_data['dir_name'].unique())
    assert (pivoted_data.index >= 0).all()
    for column in pivoted_data:
        dir_data = metric_data[metric_data['dir_name'] == column]
        dir_data = dir_data[np.isfinite(dir_data['value'])]
        dir_values = dir_data['value']
        column_data = pivoted_data[column]
        assert dir_values.iloc[0] == column_data.iloc[0]
        assert np.allclose(dir_values.loc[dir_values.last_valid_index()],
                           column_data.loc[column_data.last_valid_index()])
        assert column_data.min() >= dir_values.min()
        assert column_data.max() <= dir_values.max()
        original_wall_times = dir_data['wall_time']
        relative_wall_times = original_wall_times - original_wall_times.min()
        # Extra str rounding needed to combat floating-point inaccuracies
        rounded_relative_times = util.round_sig_figs(relative_wall_times, 3).apply(lambda x: f'{x:3g}')
        column_data = column_data[np.isfinite(column_data)]
        reindexed_pivoted_data = pd.DataFrame(column_data).set_index(
            pd.Index(
                column_data.index.to_series().apply(lambda x: f'{x:3g}')))
        unduplicated_time_mask = ~rounded_relative_times.duplicated(keep='last')
        assert np.allclose(reindexed_pivoted_data.loc[rounded_relative_times].values[unduplicated_time_mask].squeeze(),
                           dir_data['value'].values[unduplicated_time_mask])


def test_bootstrap_aggregate(pivoted_data):
    bootstrapped_mean, bootstrapped_std = plot.bootstrap_aggregate(pivoted_data, np.mean, 10)
    assert len(bootstrapped_mean) == len(pivoted_data)
    assert len(bootstrapped_std) == len(pivoted_data)
    assert bootstrapped_mean.min() >= pivoted_data.min().min()
    assert bootstrapped_mean.max() <= pivoted_data.max().max()
    assert (pivoted_data.min(axis=1) <= bootstrapped_mean).all()
    assert (bootstrapped_mean <= pivoted_data.max(axis=1)).all()
    assert (bootstrapped_std >= 0).all()
