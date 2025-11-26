import os
import warnings
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from statannotations.Annotator import Annotator
from scipy.stats import pearsonr, permutation_test, wilcoxon, false_discovery_control, bootstrap, ttest_rel


def model_post_hoc_tests(model_type, global_results_dic, data_info_file, results_save_dir):
    """
    Run and save post-hoc statistical tests and plots for a set of model outputs.

    This function iterates over entries in global_results_dic, combines each entry with
    selected columns from data_info_file, and runs a predefined set of post-hoc tests
    (or plotting helpers) for categorical and numeric covariates. Results are written
    to disk (an Excel file with the combined test data and multiple PNG figures).

    Parameters
    ----------
    model_type : str
        Short identifier for the model (used when composing output filenames).
    global_results_dic : Mapping[str, pandas.DataFrame]
        Mapping from output name -> results DataFrame. Each DataFrame should be index-aligned
        with data_info_file (or at least compatible for pandas.concat along axis=1). The
        function concatenates the DataFrame for each output with the selected columns from
        data_info_file to form the per-output stat_test_data.
    data_info_file : pandas.DataFrame
        DataFrame containing the covariates used for post-hoc testing. The function expects
        the following qualitative columns: ['Recurrence', 'Device', 'Carbidopa'] and the
        following quantitative column: ['Log10_volume']. These column names are currently
        hard-coded and must be present.
    results_save_dir : str or pathlib.Path
        Base directory where subdirectories (one per output) will be created and where the
        Excel file ('stats_test_data.xlsx') and generated PNG plots will be saved.

    Returns
    -------
    None
    """
    qualitative_tests = ['Recurrence', 'Device']
    quantitative_tests = ['Log10_volume']  # 'Seg_volume_mL'
    for output in global_results_dic:
        current_save_dir = os.path.join(results_save_dir, output)
        os.makedirs(current_save_dir, exist_ok=True)
        stat_test_data = pd.concat([global_results_dic[output], data_info_file[qualitative_tests + quantitative_tests]], axis=1)
        stat_test_data.to_excel(os.path.join(current_save_dir, 'stats_test_data.xlsx'))
        metrics_to_test = ['nnUNet_Dice_tumor', 'HM_Dice_tumor', 'HM_Surface_Dice_tumor', 'HM_Hausdorff95_tumor']
        for value in metrics_to_test:
            for qual in qualitative_tests:
                if len(stat_test_data[qual].unique()) != 1:
                    file_name = os.path.join(current_save_dir, '%s_%s_%s_vs_%s.png' % (model_type, output, qual, value))
                    post_hoc_test_factor(stat_test_data.drop(stat_test_data.index[stat_test_data[value].isna()], axis=0), 
                                         qual, value, save_path=file_name)
                    if qual != 'Recurrence':
                        file_name = os.path.join(current_save_dir, '%s_%s_Recurrence_vs_%s_hue_%s.png' % (model_type, output, value, qual))
                        post_hoc_test_factor(stat_test_data.drop(stat_test_data.index[stat_test_data[value].isna()], axis=0),
                                              'Recurrence', value, hue=qual, save_path=file_name)
            for quant in quantitative_tests:
                file_name = os.path.join(current_save_dir, '%s_%s_%s_vs_%s.png' % (model_type, output, quant, value))
                post_hoc_test_numeric(stat_test_data.drop(stat_test_data.index[stat_test_data[value].isna()], axis=0),
                                       quant, value, file_name, 'Recurrence')


def check_not_empty_series(data, x, current_x, hue=None, current_hue=None):
    """
    Check whether a filtered pandas Series (column) contains any elements.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the columns to filter.
    x : str
        Column name whose values will be checked.
    current_x : object
        Value in column `x` to select (equality comparison).
    hue : str, optional
        Optional secondary column name to further filter by. If provided,
        `current_hue` must also be provided.
    current_hue : object, optional
        Value in column `hue` to select (equality comparison). Must be provided
        when `hue` is provided.

    Returns
    -------
    bool
        True if at least one row matches the filtering criteria (i.e. the
        resulting Series is not empty); False otherwise.
    """
    if (hue is not None and current_hue is None) or (hue is None and current_hue is not None):
        raise ValueError('If one of hue or current_hue is given, the other must also be given.')
    current_data = data[x][data[x] == current_x]
    if hue is not None:
        current_data = current_data[data[hue] == current_hue]
    return len(current_data.index) != 0


def post_hoc_test_factor(data, x, y, save_path, hue=None, paired=False):
    """
    Perform pairwise post-hoc statistical tests and annotate a seaborn boxplot.

    This function draws a boxplot of `y` grouped by `x` (and optionally `hue`), computes pairwise
    comparisons between groups, applies a non-parametric statistical test for each comparison,
    annotates the plot with the test results, saves the figure to disk, and closes the figure.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe containing the columns named by `x`, `y`, and optionally `hue`.
    x : str
        Column name in `data` to use as the primary categorical factor on the x-axis.
    y : str
        Column name in `data` containing the numeric/continuous outcome to compare across groups.
    save_path : str or pathlib.Path
        Filesystem path where the resulting figure will be saved (e.g., "out/figure.png").
    hue : str, optional
        Column name in `data` used as a secondary grouping variable. If provided, pairwise
        comparisons are performed within each level of `x` across levels of `hue`. If None,
        comparisons are performed across levels of `x`. Default is None.
    paired : bool, optional
        If True, configure the Annotator to run a paired test (Wilcoxon signed-rank).
        If False, run an unpaired test (Mann–Whitney U). Default is False.

    Returns
    -------
    None
        The function saves the annotated plot to `save_path` as a side effect.
"""
    sns.set_style('ticks')
    plotting_parameters = {'data': data, 'x': x, 'y': y, 'hue': hue}
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.boxplot(ax=ax, **plotting_parameters)
    if hue is not None:
        pairs = [[(factor, hue1), (factor, hue2)] for hue1, hue2 in combinations(data[hue].unique(), 2) for factor in data[x].unique()
                 if check_not_empty_series(data, x, factor, hue, hue1) and check_not_empty_series(data, x, factor, hue, hue2)]
    else:
        pairs = [(factor1, factor2) for factor1, factor2 in combinations(data[x].unique(), 2)
                 if check_not_empty_series(data, x, factor1) and check_not_empty_series(data, x, factor2)]
    annotator = Annotator(ax, pairs, **plotting_parameters)
    if paired:
        annotator.configure(text_format='simple', show_test_name=False, test='Wilcoxon', comparisons_correction='BH', verbose=False)
    else:
        annotator.configure(text_format='simple', show_test_name=False, test='Mann-Whitney', comparisons_correction='BH', verbose=False)
    annotator.apply_test(nan_policy='omit')
    annotator.annotate()
    sns.despine()
    fig.savefig(save_path)
    plt.close(fig)
    

def post_hoc_test_numeric(data, x, y, save_path, hue=None, **kwargs):
    """
    Create and save a scatter plot with an overlaid linear regression line and display Pearson correlation.

    This function generates a scatter plot of two numeric variables from a pandas DataFrame, optionally
    coloring points by a categorical "hue" column. It overlays a linear regression line (seaborn.regplot),
    computes the Pearson correlation coefficient and two-tailed p-value for the full x/y series, annotates
    the plot with r and p, and saves the figure to disk.

    Parameters
    ----------
    data : pandas.DataFrame
        Source dataframe containing the columns referenced by x, y, and optionally hue.
    x : str
        Column name in `data` to use for the x-axis (numeric).
    y : str
        Column name in `data` to use for the y-axis (numeric).
    save_path : str or pathlib.Path
        Filesystem path where the generated figure will be saved (e.g., PNG, PDF).
    hue : str, optional
        Column name in `data` used to color/label points by group. If provided, unique values are
        mapped to a preset color list and a legend is drawn. Default is None (no grouping).
    **kwargs : dict
        Additional keyword arguments forwarded to seaborn.regplot (for example, `order=`, `ci=`,
        `line_kws=`, etc.). The function forces `color='k'` and `scatter=False` for the regplot call.

    Returns
    -------
    None
        The function saves the plot to disk and does not return a value.
    """
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(10,8))
    if hue is not None:
        color_list = ['b', 'g', 'r', 'c', 'm', 'y']
        label_mapping = {label: color_list[i] for i, label in enumerate(data[hue].unique())}
        hue_data = data[hue].replace(label_mapping)
        for label in label_mapping:
            ax.scatter(x=data[x][data[hue] == label], y=data[y][data[hue] == label], c=hue_data[data[hue] == label], label=label)
    else:
        ax.scatter(x=data[x], y=data[y])
    ax = sns.regplot(data=data, x=x, y=y, seed=111, truncate=False, color='k', scatter=False, **kwargs)
    ax.legend(title=hue, loc='upper right')
    r, p = pearsonr(data[x], data[y])
    if r >= 0:
        ax.text(0.05, 0.9, 'r=%.2f\np=%.2g' % (r, p), transform=ax.transAxes)
    else:
        ax.text(0.05, 0.1, 'r=%.2f\np=%.2g' % (r, p), transform=ax.transAxes)
    sns.despine()
    fig.savefig(save_path)
    plt.close(fig)

# TODO : modify this function according to the one in grouped_posthoc_tests.ipynb
def rank_and_compare_models(recap_results_file_dic, recap_results_file_path):
    """
    Rank and compare multiple model experiments using per-patient metric ranks, permutation tests and
    pairwise statistical comparisons.
    This function performs a series of post-hoc analyses to compare several experiments (models):
    - Computes per-patient dense ranks for many pre-defined metrics (descending for performance metrics,
        ascending for distance metrics such as Hausdorff).
    - Aggregates metric ranks to per-patient cumulative ranks and computes a Final Rank Score (FRS)
        per experiment by averaging cumulative ranks across patients. A 95% percentile bootstrap
        confidence interval for each experiment's FRS is computed.
    - Performs permutation-based rank-comparison tests between every pair of experiments using the
        per-patient cumulative ranks. P-values are saved and visualized as heatmaps; Benjamini-Hochberg
        false discovery rate correction is applied and corrected heatmaps are also saved.
    - For a selected subset of metrics (metrics_to_test), computes additional pairwise tests using:
            - permutation test of paired t-statistic (computed via ttest_rel),
            - permutation test of mean difference,
            - permutation test of median difference,
            - Wilcoxon signed-rank test.
        For each experiment and metric, bootstrap 95% CIs for mean and median are also computed.
    - Saves pairwise p-values (JSON) and p-value heatmap figures (original and FDR-corrected) to disk.
    - Updates the provided recap DataFrame dictionary in-place with computed FRS, CIs, and per-metric
        CI columns.

    Parameters
    ----------
    - recap_results_file_dic : dict[str, pandas.DataFrame]
            A mapping of grouping keys to a summary pandas DataFrame for that group. Each DataFrame should
            have experiment identifiers as its index. This DataFrame will be updated in-place with FRS and
            CI columns produced by this function.
    - recap_results_file_path : str or os.PathLike
            Path to the recap results file (used only to determine the output directory root where
            Model_ranking_comparison/ subfolders and figures/JSON are saved).
    Returns
    -------
    None
        This function modifies the input DataFrames in-place and saves additional output files.
    """
    # Model ranking https://arxiv.org/abs/1811.02629
    # From metric computed on all patients compute the rank between all tests for each metric (except detection metric) and all patients
    # Average across the individual ranks for each patients (cumulative rank), average across all patients to obtain final ranking score (FRS)
    # Do it separately for nnUNet and HM metrics. For detection metrics only average across metrics (one metric for all patient)
    # Permutation testing to determine statistical significance of the relative rankings between each pair of tests. 
    # For each pair, 100 000 random permutation of the cumulative ranks of each patient. For each permutation, compute the FRS and the difference in FRS between tests.
    # The proportion of times the difference in FRS calculated using randomly permuted data exceeded the observed difference in FRS (i.e., using the actual data) indicated the statistical significance of their relative rankings as a p-value (present in upper triangular matrix)
    # Include post hoc tests between the different tests (wilcoxon score of superiority between best and the others or each pair of tests for all continuous metric considering all patient results)

    rank_metrics_dic = {'nnUNet': ['nnUNet_Dice', 'nnUNet_Accuracy', 'nnUNet_Sensitivity', 'nnUNet_Specificity', 'nnUNet_PPV', 'nnUNet_NPV'], 
                        'HM': ['HM_Dice', 'HM_Surface_Dice', 'HM_Hausdorff95'], 
                        'HM_BRATS_like': ['HM_Dice', 'HM_Hausdorff95', 'HM_Sensitivity', 'HM_Specificity'], 
                        'HM_Dice_only': ['HM_Dice'], 
                        'HM_Surface_Dice_only': ['HM_Surface_Dice']}
        
    metrics_to_test = {'nnUNet': ['nnUNet_Dice'], 'HM': ['HM_Dice', 'HM_Surface_Dice', 'HM_Hausdorff95']}
    
    for key in recap_results_file_dic:
        save_root = os.path.join(os.path.dirname(recap_results_file_path), 'Model_ranking_comparison', key)
        split = key.split('_')[0]
        output = '_'.join(key.split('_')[1:])
        # Load global results of each experiment
        global_results = {}
        for experiment in recap_results_file_dic[key].index:
            global_results[experiment] = pd.read_excel(os.path.join(os.environ.get('nnUNet_results_analysis'),
                                                                    '%s_results' % experiment.split('__')[0],
                                                                    '__'.join(experiment.split('__')[1:]), 
                                                                    'Global_results_%s.xlsx' % split.lower()),
                                                       sheet_name='%s_Sample' % output, index_col=0)
            
        # For each metric get the rank of each experiment
        # At the same time, store the metrics results we need later 
        # /!\ this only work if keys of metrics_to_test are subset of rank_metrics_dic and that for each key the values are a subset of the values with the same key in rank_metrics_dic
        # TODO : make it work even if not subset
        if not set(metrics_to_test.keys()).issubset(set(rank_metrics_dic.keys())):
            warnings.warn("metric_to_test keys is not a subset of rank_metrics_dic which is not currently supported. "
                         "Following keys will not be taken into account for performance testing: %s." % set(metrics_to_test.keys()).difference(set(set(rank_metrics_dic.keys()))))
        global_results_by_metric = {}
        rank_results_by_metric = {}
        for metric_type in rank_metrics_dic:
            if metric_type in metrics_to_test:
                global_results_by_metric[metric_type] = {}
                if not set(metrics_to_test[metric_type]).issubset(set(rank_metrics_dic[metric_type])):
                    warnings.warn("metric_to_test keys for %s is not a subset of rank_metrics_dic for %s which is not currently supported. "
                                "Following keys will not be taken into account for performance testing: %s." % 
                                (metric_type, metric_type, set(metrics_to_test[metric_type].keys()).difference(set(set(rank_metrics_dic[metric_type].keys())))))
            rank_results_by_metric[metric_type] = {}
            for metric in rank_metrics_dic[metric_type]:
                labels_col = set([col for experiment in global_results for col in global_results[experiment] if col.startswith(metric)])
                for col in labels_col:
                    label = col.split(metric + '_')[-1]
                    if metric_type in metrics_to_test and metric in metrics_to_test[metric_type] and label not in global_results_by_metric[metric_type]:
                        global_results_by_metric[metric_type][label] = {}
                    if label not in rank_results_by_metric[metric_type]:
                        rank_results_by_metric[metric_type][label] = {}
                    experiment_list = [experiment for experiment in global_results if col in global_results[experiment]]
                    metric_results = pd.concat([global_results[experiment][col] for experiment in experiment_list],
                                               axis=1, keys=experiment_list)
                    if metric_type in metrics_to_test and metric in metrics_to_test[metric_type]:
                        global_results_by_metric[metric_type][label][col] = metric_results
                    rank_results_by_metric[metric_type][label][col] = metric_results.rank(axis=1, method='dense', na_option='keep',
                                                                                          ascending=False if not 'Hausdorff' in metric else True)

        # Average across the individual metric ranks for each patients (cumulative rank) and average of cumulative rank (final rank score)
        # We also compute 95% confidence interval on final rank score
        # TODO : we could try vectorization also here but it's already pretty quick
        cumulative_ranks = {}
        for metric_type in rank_results_by_metric:
            cumulative_ranks[metric_type] = {}
            for label in rank_results_by_metric[metric_type]:
                cumulative_ranks[metric_type][label] = {}
                for experiment in global_results:
                    experiment_ranks = {metric: rank_results_by_metric[metric_type][label][metric][experiment] for metric in rank_results_by_metric[metric_type][label]
                                        if experiment in rank_results_by_metric[metric_type][label][metric]}
                    if experiment_ranks:
                        expirement_ranks = pd.concat(experiment_ranks, axis=1)
                        cumulative_ranks[metric_type][label][experiment] = expirement_ranks.mean(axis=1, skipna=True)
                        # Compute final rank score for each experiment
                        recap_results_file_dic[key].loc[experiment, 'FRS_%s_%s' % (metric_type, label)] = cumulative_ranks[metric_type][label][experiment].mean(skipna=True)
                        # Compute 95% CI
                        ranks_ci = bootstrap((cumulative_ranks[metric_type][label][experiment],), np.mean, n_resamples=10000, batch=1000, 
                                             confidence_level=0.95, method='percentile').confidence_interval
                        recap_results_file_dic[key].loc[experiment, 'FRS_%s_%s_low_95CI' % (metric_type, label)] = ranks_ci.low
                        recap_results_file_dic[key].loc[experiment, 'FRS_%s_%s_high_95CI' % (metric_type, label)] = ranks_ci.high
                    else:
                        recap_results_file_dic[key].loc[experiment, 'FRS_%s_%s' % (metric_type, label)] = np.nan        

        # Rank permutation tests (rank permutation testing and wilcoxon on performance)
        # TODO : combine permutation_test for metric_type + label + combinations in one run ? 
        rank_permutation_save_root = os.path.join(save_root, 'Rank_permutation_tests')
        for metric_type in cumulative_ranks:
            current_rank_permutation_save_dir = os.path.join(rank_permutation_save_root, metric_type)
            os.makedirs(current_rank_permutation_save_dir, exist_ok=True)
            for label in cumulative_ranks[metric_type]:
                if len(cumulative_ranks[metric_type][label].keys()) > 1:
                    # We can't load pvalues previously computed as ranks will change each time we add an experiment, ranks are dynamic
                    # We must recompute everything at each new experiment
                    experiment_combinations_list = list(combinations(list(cumulative_ranks[metric_type][label].keys()), 2))
                    all_exp1 = np.array([cumulative_ranks[metric_type][label][exp1].to_numpy() for exp1, exp2 in experiment_combinations_list])
                    all_exp2 = np.array([cumulative_ranks[metric_type][label][exp2].to_numpy() for exp1, exp2 in experiment_combinations_list])
                    all_pval_rank = permutation_test((all_exp1, all_exp2), lambda x,y,axis:  np.nanmean(x - y, axis=axis), permutation_type='samples',
                                                    n_resamples=100000, batch=1000, alternative='less', random_state=123, vectorized=True, axis=1).pvalue
                    permutation_rank_tests_dic = {}
                    for i, (exp1, exp2) in enumerate(experiment_combinations_list):
                        if exp1 not in permutation_rank_tests_dic:
                            permutation_rank_tests_dic[exp1] = {exp1: 0.5}
                        if exp2 not in permutation_rank_tests_dic:
                            permutation_rank_tests_dic[exp2] = {exp2: 0.5}
                        permutation_rank_tests_dic[exp1][exp2] = all_pval_rank[i]
                        permutation_rank_tests_dic[exp2][exp1] = 1 - all_pval_rank[i] 
                    permutation_rank_tests_df = pd.DataFrame(permutation_rank_tests_dic)
                    pval_heatmap(recap_results_file_dic[key]['FRS_%s_%s' % (metric_type, label)], permutation_rank_tests_df, 
                                os.path.join(current_rank_permutation_save_dir, '%s_%s.png' % (metric_type, label)), mask_diag=True, sort_values=True,
                                ci = (recap_results_file_dic[key]['FRS_%s_%s_low_95CI' % (metric_type, label)], recap_results_file_dic[key]['FRS_%s_%s_high_95CI' % (metric_type, label)]))
                    permutation_rank_tests_df[:] = false_discovery_control(permutation_rank_tests_df.to_numpy().flatten(),
                                                                            axis=None, method='bh').reshape(permutation_rank_tests_df.shape)
                    pval_heatmap(recap_results_file_dic[key]['FRS_%s_%s' % (metric_type, label)], permutation_rank_tests_df, 
                                os.path.join(current_rank_permutation_save_dir, '%s_%s_pvalcorr.png' % (metric_type, label)), mask_diag=True, sort_values=True,
                                ci = (recap_results_file_dic[key]['FRS_%s_%s_low_95CI' % (metric_type, label)], recap_results_file_dic[key]['FRS_%s_%s_high_95CI' % (metric_type, label)]))
            
        # Direct test on difference of performance (permutation t test and wilcoxon)
        ttest_permutation_save_root = os.path.join(save_root, 'T_test_permutation_tests')
        wilcoxon_save_root = os.path.join(save_root, 'Wilcoxon_tests')
        mean_diff_permutation_save_root = os.path.join(save_root, 'Mean_diff_permutation_tests')
        median_diff_permutation_save_root = os.path.join(save_root, 'Median_diff_permutation_tests')
        for metric_type in global_results_by_metric:
            current_ttest_permutation_save_dir = os.path.join(ttest_permutation_save_root, metric_type)
            os.makedirs(current_ttest_permutation_save_dir, exist_ok=True)
            current_wilcoxon_save_dir = os.path.join(wilcoxon_save_root, metric_type)
            os.makedirs(current_wilcoxon_save_dir, exist_ok=True)
            current_mean_diff_permutation_save_dir = os.path.join(mean_diff_permutation_save_root, metric_type)
            os.makedirs(current_mean_diff_permutation_save_dir, exist_ok=True)
            current_median_diff_permutation_save_dir = os.path.join(median_diff_permutation_save_root, metric_type)
            os.makedirs(current_median_diff_permutation_save_dir, exist_ok=True)
            for label in global_results_by_metric[metric_type]:
                for metric in global_results_by_metric[metric_type][label]:
                    # TODO : should this be moved to evaluation_utils.extract_fold_results ?
                    # TODO : vectorize this also?
                    # Compute 95% CIs of the metrics here
                    for exp in global_results_by_metric[metric_type][label][metric].keys():
                        mean_metric_ci = bootstrap((global_results_by_metric[metric_type][label][metric][exp],), np.nanmean, n_resamples=10000, batch=1000,  
                                              confidence_level=0.95, method='percentile').confidence_interval
                        recap_results_file_dic[key].loc[exp, 'Mean overall %s_low_95CI' % metric] = mean_metric_ci.low
                        recap_results_file_dic[key].loc[exp, 'Mean overall %s_high_95CI' % metric] = mean_metric_ci.high
                        median_metric_ci = bootstrap((global_results_by_metric[metric_type][label][metric][exp],), np.nanmedian, n_resamples=10000, batch=1000,  
                                              confidence_level=0.95, method='percentile').confidence_interval
                        recap_results_file_dic[key].loc[exp, 'Median overall %s_low_95CI' % metric] = median_metric_ci.low
                        recap_results_file_dic[key].loc[exp, 'Median overall %s_high_95CI' % metric] = median_metric_ci.high
                    if any([metric.startswith(_) for _ in metrics_to_test[metric_type]]) and len(global_results_by_metric[metric_type][label][metric].keys()) > 1:
                        # Here we can save and load pvalues to gain time are differences in performance between data are supposed to be static
                        if os.path.exists(os.path.join(current_ttest_permutation_save_dir, 'pairwise_pvalues_%s.json' % metric)):
                            permutation_t_tests_dic = load_json(os.path.join(current_ttest_permutation_save_dir, 'pairwise_pvalues_%s.json' % metric))
                            wilcox_test_dic = load_json(os.path.join(current_wilcoxon_save_dir, 'pairwise_pvalues_%s.json' % metric))
                            permutation_mean_diff_dic = load_json(os.path.join(current_mean_diff_permutation_save_dir, 'pairwise_pvalues_%s.json' % metric))
                            permutation_median_diff_dic = load_json(os.path.join(current_median_diff_permutation_save_dir, 'pairwise_pvalues_%s.json' % metric))
                            if set(permutation_t_tests_dic.keys()) == set(wilcox_test_dic.keys()) and set(permutation_t_tests_dic.keys()) == set(permutation_mean_diff_dic.keys()) and set(permutation_t_tests_dic.keys()) == set(permutation_median_diff_dic.keys()):
                                for exp in permutation_t_tests_dic:
                                    if set(permutation_t_tests_dic[exp].keys()) != set(wilcox_test_dic[exp].keys()) or set(permutation_t_tests_dic[exp].keys()) != set(permutation_t_tests_dic[exp].keys()) or set(permutation_t_tests_dic[exp].keys()) != set(permutation_median_diff_dic[exp].keys()):
                                        permutation_t_tests_dic = {}
                                        wilcox_test_dic = {}
                                        permutation_mean_diff_dic = {}
                                        permutation_median_diff_dic = {}
                                        break
                            else:
                                permutation_t_tests_dic = {}
                                wilcox_test_dic = {}
                                permutation_mean_diff_dic = {}
                                permutation_median_diff_dic = {}
                        else:
                            permutation_t_tests_dic = {}
                            wilcox_test_dic = {}
                            permutation_mean_diff_dic = {}
                            permutation_median_diff_dic = {}
                        experiment_combinations_list = list(combinations(list(global_results_by_metric[metric_type][label][metric].keys()), 2))
                        for exp1 in permutation_t_tests_dic:
                            for exp2 in permutation_t_tests_dic[exp1]:
                                if (exp1, exp2) in experiment_combinations_list:
                                    experiment_combinations_list.remove((exp1, exp2))
                        all_exp1 = np.array([global_results_by_metric[metric_type][label][metric][exp1].to_numpy() for exp1, exp2 in experiment_combinations_list])
                        all_exp2 = np.array([global_results_by_metric[metric_type][label][metric][exp2].to_numpy() for exp1, exp2 in experiment_combinations_list])
                        all_pval_ttest = permutation_test((all_exp1, all_exp2), lambda x,y,axis:  ttest_rel(x,y,axis=axis,nan_policy='omit').statistic, 
                                                          permutation_type='samples', n_resamples=100000, batch=1000, vectorized=True, axis=1,
                                                          alternative='greater' if not 'Hausdorff' in metric else 'less', random_state=123).pvalue
                        all_pval_mean_diff = permutation_test((all_exp1, all_exp2), lambda x,y,axis:  np.nanmean(x - y, axis=axis), 
                                                              permutation_type='samples', n_resamples=100000, batch=1000, vectorized=True, axis=1, 
                                                              alternative='greater' if not 'Hausdorff' in metric else 'less', random_state=123).pvalue
                        all_pval_median_diff = permutation_test((all_exp1, all_exp2), lambda x,y,axis:  np.nanmedian(x - y, axis=axis), 
                                                                permutation_type='samples', n_resamples=100000, batch=1000, vectorized=True, axis=1, 
                                                                alternative='greater' if not 'Hausdorff' in metric else 'less', random_state=123).pvalue
                        try:
                            all_pval_wilcox = wilcoxon(all_exp1, all_exp2, axis=1, 
                                                    alternative='greater' if not 'Hausdorff' in metric else 'less', nan_policy='omit').pvalue
                        except:
                            all_pval_wilcox = []
                            for exp1, exp2 in experiment_combinations_list:
                                if np.all(global_results_by_metric[metric_type][label][metric][exp1] == global_results_by_metric[metric_type][label][metric][exp2]):
                                    pval_wilcox = 1
                                else:
                                    try:
                                        pval_wilcox = wilcoxon(global_results_by_metric[metric_type][label][metric][exp1], global_results_by_metric[metric_type][label][metric][exp2],
                                                            alternative='greater' if not 'Hausdorff' in metric else 'less', nan_policy='omit').pvalue
                                    except:
                                        # In case error because all diff are zero even though it should be handled above
                                        pval_wilcox = wilcoxon(global_results_by_metric[metric_type][label][metric][exp1], global_results_by_metric[metric_type][label][metric][exp2],
                                                            alternative='greater' if not 'Hausdorff' in metric else 'less', nan_policy='omit', zero_method='zsplit').pvalue
                                all_pval_wilcox.append(pval_wilcox)
                            all_pval_wilcox = np.array(all_pval_wilcox)
                                
                        for i, (exp1, exp2) in enumerate(experiment_combinations_list):
                            if exp1 not in permutation_t_tests_dic:
                                permutation_t_tests_dic[exp1] = {exp1: 0.5}
                                wilcox_test_dic[exp1] = {exp1: 0.5}
                                permutation_mean_diff_dic[exp1] = {exp1: 0.5}
                                permutation_median_diff_dic[exp1] = {exp1: 0.5}
                            if exp2 not in permutation_t_tests_dic:
                                permutation_t_tests_dic[exp2] = {exp2: 0.5}
                                wilcox_test_dic[exp2] = {exp2: 0.5}
                                permutation_mean_diff_dic[exp2] = {exp2: 0.5}
                                permutation_median_diff_dic[exp2] = {exp2: 0.5}
                            permutation_t_tests_dic[exp1][exp2] = all_pval_ttest[i]
                            wilcox_test_dic[exp1][exp2] = all_pval_wilcox[i]
                            permutation_mean_diff_dic[exp1][exp2] = all_pval_mean_diff[i]
                            permutation_median_diff_dic[exp1][exp2] = all_pval_median_diff[i]
                            permutation_t_tests_dic[exp2][exp1] = 1 - all_pval_ttest[i]
                            wilcox_test_dic[exp2][exp1] = 1 - all_pval_wilcox[i]
                            permutation_mean_diff_dic[exp2][exp1] = 1 - all_pval_mean_diff[i]
                            permutation_median_diff_dic[exp2][exp1] = 1 - all_pval_median_diff[i]                            
                            
                        save_json(permutation_t_tests_dic, os.path.join(current_ttest_permutation_save_dir, 'pairwise_pvalues_%s.json' % metric))
                        save_json(wilcox_test_dic, os.path.join(current_wilcoxon_save_dir, 'pairwise_pvalues_%s.json' % metric))
                        save_json(permutation_mean_diff_dic, os.path.join(current_mean_diff_permutation_save_dir, 'pairwise_pvalues_%s.json' % metric))
                        save_json(permutation_median_diff_dic, os.path.join(current_median_diff_permutation_save_dir, 'pairwise_pvalues_%s.json' % metric))
                        
                        permutation_t_tests_df = pd.DataFrame(permutation_t_tests_dic)
                        pval_heatmap(recap_results_file_dic[key]['Mean overall %s' % metric], permutation_t_tests_df, 
                                    os.path.join(current_ttest_permutation_save_dir, '%s.png' % metric), mask_diag=True, sort_values=True,
                                    ci = (recap_results_file_dic[key]['Mean overall %s_low_95CI' % metric], recap_results_file_dic[key]['Mean overall %s_high_95CI' % metric]))
                        permutation_t_tests_df[:] = false_discovery_control(permutation_t_tests_df.to_numpy().flatten(),
                                                                            axis=None, method='bh').reshape(permutation_t_tests_df.shape)
                        pval_heatmap(recap_results_file_dic[key]['Mean overall %s' % metric], permutation_t_tests_df, 
                                    os.path.join(current_ttest_permutation_save_dir, '%s_pvalcorr.png' % metric), mask_diag=True, sort_values=True,
                                    ci = (recap_results_file_dic[key]['Mean overall %s_low_95CI' % metric], recap_results_file_dic[key]['Mean overall %s_high_95CI' % metric]))
                        wilcox_test_df = pd.DataFrame(wilcox_test_dic)
                        pval_heatmap(recap_results_file_dic[key]['Median overall %s' % metric], wilcox_test_df, 
                                    os.path.join(current_wilcoxon_save_dir, '%s.png' % metric), mask_diag=True, sort_values=True,
                                    ci = (recap_results_file_dic[key]['Median overall %s_low_95CI' % metric], recap_results_file_dic[key]['Median overall %s_high_95CI' % metric]))
                        wilcox_test_df[:] = false_discovery_control(wilcox_test_df.to_numpy().flatten(),
                                                                    axis=None, method='bh').reshape(wilcox_test_df.shape)
                        pval_heatmap(recap_results_file_dic[key]['Median overall %s' % metric], wilcox_test_df, 
                                    os.path.join(current_wilcoxon_save_dir, '%s_pvalcorr.png' % metric), mask_diag=True, sort_values=True,
                                    ci = (recap_results_file_dic[key]['Median overall %s_low_95CI' % metric], recap_results_file_dic[key]['Median overall %s_high_95CI' % metric]))
                        permutation_mean_diff_df = pd.DataFrame(permutation_mean_diff_dic)
                        pval_heatmap(recap_results_file_dic[key]['Mean overall %s' % metric], permutation_mean_diff_df, 
                                    os.path.join(current_mean_diff_permutation_save_dir, '%s.png' % metric), mask_diag=True, sort_values=True,
                                    ci = (recap_results_file_dic[key]['Mean overall %s_low_95CI' % metric], recap_results_file_dic[key]['Mean overall %s_high_95CI' % metric]))
                        permutation_mean_diff_df[:] = false_discovery_control(permutation_mean_diff_df.to_numpy().flatten(),
                                                                            axis=None, method='bh').reshape(permutation_mean_diff_df.shape)
                        pval_heatmap(recap_results_file_dic[key]['Mean overall %s' % metric], permutation_mean_diff_df, 
                                    os.path.join(current_mean_diff_permutation_save_dir, '%s_pvalcorr.png' % metric), mask_diag=True, sort_values=True,
                                    ci = (recap_results_file_dic[key]['Mean overall %s_low_95CI' % metric], recap_results_file_dic[key]['Mean overall %s_high_95CI' % metric]))
                        permutation_median_diff_df = pd.DataFrame(permutation_median_diff_dic)
                        pval_heatmap(recap_results_file_dic[key]['Median overall %s' % metric], permutation_median_diff_df, 
                                    os.path.join(current_median_diff_permutation_save_dir, '%s.png' % metric), mask_diag=True, sort_values=True,
                                    ci = (recap_results_file_dic[key]['Median overall %s_low_95CI' % metric], recap_results_file_dic[key]['Median overall %s_high_95CI' % metric]))
                        permutation_median_diff_df[:] = false_discovery_control(permutation_median_diff_df.to_numpy().flatten(),
                                                                            axis=None, method='bh').reshape(permutation_median_diff_df.shape)
                        pval_heatmap(recap_results_file_dic[key]['Median overall %s' % metric], permutation_median_diff_df, 
                                    os.path.join(current_median_diff_permutation_save_dir, '%s_pvalcorr.png' % metric), mask_diag=True, sort_values=True,
                                    ci = (recap_results_file_dic[key]['Median overall %s_low_95CI' % metric], recap_results_file_dic[key]['Median overall %s_high_95CI' % metric]))


def pval_heatmap(values, pvals, save_path, mask_diag=False, sort_values=True, ci=None):
    """
    Generate and save a heatmap visualization of pairwise p-values with optional test-value labels
    and confidence-interval annotations.

    Parameters
    ----------
    values : pandas.Series
        1-D series of numeric summary values (e.g. test statistics or effect sizes) indexed by test
        identifiers. If sort_values is True, this series will be dropna'd and sorted and the p-value
        matrix (and confidence intervals, if provided) will be reindexed to match the sorted order.
    pvals : pandas.DataFrame
        Square DataFrame of pairwise p-values. Rows and columns must be indexed by the same test
        identifiers as `values`. The shape of this DataFrame determines the heatmap dimensions.
    save_path : str or os.PathLike
        File path where the heatmap image will be saved. A text file with label mappings will be
        written alongside the image if `ci` is provided (same base name, extension ".txt").
    mask_diag : bool, optional (default=False)
        If True, the diagonal cells of the p-value matrix will be masked (hidden) in the heatmap.
    sort_values : bool, optional (default=True)
        If True, `values` will be dropna'd and sorted; `pvals` (and `ci` if provided) will be
        reordered to match that sorted index. If False, the original ordering is preserved.
    ci : None or tuple(list_like, list_like), optional (default=None)
        Optional pair (lower_bounds, upper_bounds) giving confidence-interval endpoints for each test.
        Each element of the tuple must be indexable by the same test identifiers as `values`.
        If provided, the printed labels include the value and its confidence interval, e.g.
        "Test0 (0.123 [0.050; 0.196])".
        If not None, `ci` must be of length 2 otherwise a ValueError is raised.

    Returns
    -------
    None
        The function saves a PNG (or other format inferred from save_path extension) heatmap to
        `save_path` and closes the figure. When `ci` is provided, a text file containing label
        mappings is also written beside the image.
    """
    if ci is not None and len(ci) != 2:
        raise ValueError('ci need to have a length of 2 with list of lower and higher confidence interval got %d' % len(ci))
    if sort_values:
        values = values.dropna().sort_values()
        pvals = pvals.loc[values.index, values.index]
        if ci is not None:
            ci = (ci[0][values.index], ci[1][values.index])
    if ci is not None:
        labels_with_vals = {_ :'Test%d (%.3f [%.3f; %.3f])' % (i, values[_], ci[0][_], ci[1][_]) for i, _ in enumerate(values.index)}
    else:
        labels_with_vals = {_ :'Test%d (%.3f)' % (i, values[_]) for i, _ in enumerate(values.index)}
    fig, ax = plt.subplots(figsize=(max(10, pvals.shape[0]), max(10, pvals.shape[0])))
    ax = sns.heatmap(pvals, cmap='bwr', vmin=0, vmax=1, center=0.5, annot=True, fmt='.3f', cbar=True,
                     xticklabels=labels_with_vals.values(), yticklabels=labels_with_vals.values(),
                     mask=np.eye(pvals.shape[0]).astype(bool) if mask_diag else False, ax=ax)
    with open(os.path.splitext(save_path)[0] + '.txt', 'w') as f:
        if ci is not None:
            f.write('\n'.join(['%s --> %s' %(value, key) for key, value in labels_with_vals.items()]))
    sns.despine()
    fig.savefig(save_path)
    plt.close(fig)