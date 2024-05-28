import json
import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import pandas as pd

import plot
import util


CORE_ALGORITHMS = ('SGD',
                   'SGDmwd',
                   'Adam',
                   'Adam_NoHPO',
                   # 'AdamQLR_Damped',
                   'AdamQLR_Damped_NoLRClipping',
                   # 'AdamQLR_NoHPO',
                   'AdamQLR_NoHPO_NoLRClipping',
                   'KFAC',
                   # 'KFAC_NoHPO',
                   'KFAC_BigBatch_NoHPO',
                   'Adam_TunedEpsilon',
                   'KFAC_Unadaptive',
                   'BaydinSGD',
                   )


def tight_savefig(directory):
    plt.savefig(directory,
                bbox_inches='tight',
                pad_inches=0)
    plt.close()


@contextmanager
def paper_theme():
    with plt.style.context('Solarize_Light2'):
        yield
        plt.gcf().set_facecolor('white')
        plt.gcf().set_size_inches(6, 4)

def rosenbrock_trajectory_plot():
    with plot.inhibit_plt_show(), paper_theme():
        plot.plot_rosenbrock_paths(
            '/scratch/dir/ImprovingKFAC/2023-09-26 Rosenbrock NoLRClipping',
            ['GD', 'GDmwd', 'Adam',
             'AdamQLR_Damped_Hessian_NoLRClipping', 'AdamQLR_Damped_Hessian_NoLRClipping_NoHPO',
             'KFAC'])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    tight_savefig('./plots/paper_ICML_Rebuttals/RosenbrockTrajectory.pdf')


def loss_evolution_plots():
    source_directories = dict(
        default=(
            '2023-09-24 UCI_Energy NoLRClipping',
            '2023-09-24 UCI_Protein NoLRClipping',
            '2023-09-24 Fashion-MNIST NoLRClipping',
            '2023-09-24 SVHN NoLRClipping',
            '2023-09-21 CIFAR-10 NoLRClipping',
            '2023-11-14 PennTreebank_GPT2_Reset',
        ),
        ASHA_Time_Training=(
            '2023-09-27 UCI_Energy ASHA_Time_Training',
            '2023-09-27 UCI_Protein ASHA_Time_Training',
            '2023-09-27 Fashion-MNIST ASHA_Time_Training',
            '2023-09-27 SVHN ASHA_Time_Training',
            '2023-09-27 CIFAR-10 ASHA_Time_Training',
        ),
        ASHA_Time_Validation=(
            '2023-09-27 UCI_Energy ASHA_Time_Validation',
            '2023-09-27 UCI_Protein ASHA_Time_Validation',
            '2023-09-27 Fashion-MNIST ASHA_Time_Validation',
            '2023-09-27 SVHN ASHA_Time_Validation',
            '2023-09-27 CIFAR-10 ASHA_Time_Validation',
        )
    )
    for result_set, result_directories in source_directories.items():
        plotted_data = {}
        for directory in result_directories:
            dataset_name = raw_dataset_name = directory.split(' ')[1]
            plotted_data[raw_dataset_name] = {}
            match raw_dataset_name:
                case 'UCI_Energy': break_point=300
                case 'UCI_Protein': break_point=300
                case 'Fashion-MNIST': break_point = 30
                case 'SVHN': break_point=250
                case 'CIFAR-10': break_point=2000
                case 'PennTreebank_GPT2_Reset': break_point=4000
            valid_metrics = ['Loss/Training',
                            'Loss/Test',
                            'Adaptive/Learning_Rate',]
            if not (dataset_name.startswith('UCI') or raw_dataset_name.startswith('PennTreebank')):
                valid_metrics.extend(['Accuracy/Training',
                                      'Accuracy/Test'])
            if 'ASHA_Time_' in directory:
                dataset_name += directory.split(' ')[-1]
            for metric in valid_metrics:
                with plot.inhibit_plt_show(), paper_theme():
                    axes, plotted_data[raw_dataset_name][metric] = plot.plot_best_run_envelopes(
                        f'/scratch/dir/ImprovingKFAC/{directory}',
                        metric,
                        log_x_axis=False,
                        included_algorithms=CORE_ALGORITHMS,
                        aggregation='median',
                        break_x_axis=break_point,
                        return_data=True)
                    if (dataset_name.startswith('UCI_Protein')
                        and metric.startswith('Loss')):
                        axes[0].set_ylim(1e-1, 1e-0)
                    if metric.startswith('Accuracy'):
                        axes[0].set_yscale('linear')
                        axes[0].set_ylim(0, 1.0)
                    # if (dataset_name.startswith('Fashion-MNIST')
                    #     and metric.startswith('Loss/')):
                    #     # 2e-6
                    #     axes[0].set_ylim(None, 7)
                    if (dataset_name.startswith('UCI_Energy')
                        and metric.startswith('Loss/')):
                        axes[0].set_ylim(8e-5, 2)
                    if (dataset_name.startswith('UCI_Energy')
                        and metric.startswith('Adaptive/Learning_Rate')):
                        axes[0].set_ylim(1e-4, 1e2)
                    if (dataset_name.startswith('CIFAR-10')
                        and result_set.startswith('ASHA')
                        and metric.startswith('Loss/Test')):
                        axes[0].set_ylim(1e-1, 1e3)
                    if (dataset_name.startswith('Fashion-MNIST')
                        and result_set.startswith('ASHA')
                        and metric.startswith('Loss')):
                        if metric == 'Loss/Training':
                            axes[0].set_ylim(1e-7, 1e1)
                        else:
                            axes[0].set_ylim(None, 1e1)
                metric_name = metric.split('/')[1]
                if metric.startswith('Loss'):
                    suffix = ' Loss'
                elif metric.startswith('Accuracy'):
                    suffix = ' Accuracy'
                else:
                    suffix = ''
                pretty_metric_name = ' '.join(metric_name.split('_'))
                axes[0].set_ylabel(f'{pretty_metric_name}{suffix}')
                tight_savefig(f'./plots/paper_ICML_Rebuttals/{dataset_name}_{metric_name}{suffix.split(" ")[-1]}.pdf')
                plt.close()
        loss_evolution_tables(result_set, plotted_data)


def loss_evolution_tables(suite, plotted_data):
    metrics = ('Loss/Training',
               'Accuracy/Training',
               'Loss/Test',
               'Accuracy/Test')
    with open(f'./plots/paper_ICML_NoClipping/FinalLosses_{suite}.tex', 'w') as table:
        for dataset_name, dataset_data in plotted_data.items():
            table.write(r'\midrule' + '\n')
            num_algorithms = len(dataset_data[metrics[0]])
            table.write(r'\multirow{'+ str(num_algorithms) + r'}{*}' + f'{{{dataset_name}}}\n')

            for algorithm in dataset_data[metrics[0]]:
                table.write('& ' + plot.KEY_TO_LABEL[algorithm] + ' \t& ')

                for metric in metrics:
                    if metric not in dataset_data:
                        table.write(r'\multicolumn{2}{c}{---}' + '\t& ')
                        continue
                    value, error = util.format_value_and_error(
                        dataset_data[metric][algorithm]["mean"].iloc[-1],
                        dataset_data[metric][algorithm]["std"].iloc[-1],
                        error_sig_figs=2)
                    table.write(f'{value} & ')
                    table.write(r'$\pm$ \num{' + f'{error}' + '} \t& ')

                generalisation_gap_mean = (dataset_data['Loss/Test'][algorithm]['mean'].iloc[-1]
                                            - dataset_data['Loss/Training'][algorithm]['mean'].iloc[-1])
                generalisation_gap_std = (dataset_data['Loss/Test'][algorithm]['std'].iloc[-1]
                                          + dataset_data['Loss/Training'][algorithm]['std'].iloc[-1])
                value, error = util.format_value_and_error(
                    generalisation_gap_mean,
                    generalisation_gap_std,
                    error_sig_figs=2)
                table.write(f'{value} & ')
                table.write(r'$\pm$ \num{' + f'{error}' + '} \t& ')

                value, error = util.format_value_and_error(
                    int(dataset_data["Loss/Training"][algorithm]["steps_mean"]),
                    dataset_data["Loss/Training"][algorithm]["steps_std"],
                    error_sig_figs=2)
                table.write(f'{value} & ')
                table.write(r'$\pm$ \num{' + f'{error}' + '} \t& ')

                value, error = util.format_value_and_error(
                    dataset_data["Loss/Training"][algorithm]["times_mean"],
                    dataset_data["Loss/Training"][algorithm]["times_std"],
                    error_sig_figs=2)
                table.write(f'{value} & ')
                table.write(r'$\pm$ \num{' + f'{error}' + r'} \\' + '\n')
        table.write(r'\bottomrule')


def sensitivity_plots():
    directories = (
        # AdamQLR without LR clipping
        '/scratch/dir/ImprovingKFAC/ray/2024-01-27T19:00:58.457718__fashion_mnist__AdamQLR_Damped_NoLRClipping__ASHA Sensitivity_Amplification',
        '/scratch/dir/ImprovingKFAC/ray/2024-01-27T19:00:58.457718__fashion_mnist__AdamQLR_Damped_NoLRClipping__ASHA Sensitivity_BatchSize',
        '/scratch/dir/ImprovingKFAC/ray/2024-01-27T19:00:58.457718__fashion_mnist__AdamQLR_Damped_NoLRClipping__ASHA Sensitivity_InitialDamping',
        '/scratch/dir/ImprovingKFAC/ray/2024-01-27T19:00:58.457718__fashion_mnist__AdamQLR_Damped_NoLRClipping__ASHA Sensitivity_SteppingFactor'
        # ICML Rebuttals
        '/scratch/dir/ImprovingKFAC/ray/2023-05-04T01:26:09.850870__fashion_mnist__KFAC__ASHA2 Sensitivity_BatchSize',
        '/scratch/dir/ImprovingKFAC/ray/2023-05-04T01:26:09.850870__fashion_mnist__KFAC__ASHA2 Sensitivity_InitialDamping',
        '/scratch/dir/ImprovingKFAC/ray/2023-05-03T23:13:22.030159__fashion_mnist__Adam__ASHA2 Sensitivity_BatchSize',
        '/scratch/dir/ImprovingKFAC/ray/2023-05-03T23:13:22.030159__fashion_mnist__Adam__ASHA2 Sensitivity_Base10_LearningRate',
    )
    for directory in directories:
        ablation_type = directory.split('_')[-1]
        algorithm_name = directory.split('__')[2]
        for metric in ('Loss/Training', 'Loss/Test'):
            with (plot.inhibit_plt_show(), paper_theme()):
                axes = plot.plot_ablation_trends(directory,
                                                 metric,
                                                 log_x_axis=False,
                                                 aggregation='median',
                                                 break_x_axis=14 if ablation_type == 'BatchSize' else False)
            loss_name = metric[5:]
            axes[0].set_ylabel(f'{loss_name} Loss')
            if algorithm_name.startswith('AdamQLR'):
                if ablation_type != 'BatchSize':
                    plt.xlim(0, 14)
                else:
                    plt.ylim(0.1, 7)
                if ablation_type == 'Amplification':
                    plt.ylim(1e-1, 1e1)
            tight_savefig(f'./plots/paper_ICML_Rebuttals/{algorithm_name}_Sensitivity_{ablation_type}_{loss_name}Loss.pdf')

    for metric in ('Loss/Training', 'Loss/Test'):
        with plot.inhibit_plt_show(), paper_theme():
            axes = plot.plot_best_run_envelopes(
                '/scratch/dir/ImprovingKFAC/2023-09-24 Fashion-MNIST NoLRClipping',
                metric,
                log_x_axis=False,
                included_algorithms=CORE_ALGORITHMS,
                aggregation='median',
                break_x_axis=14)
        loss_name = metric[5:]
        axes[0].set_ylabel(f'{loss_name} Loss')
        tight_savefig(f'./plots/paper_ICML_Rebuttals/{algorithm_name}_Sensitivity_ReprisedFashion-MNIST_{loss_name}Loss.pdf')


def hyperparameter_table(suite):
    match suite:
        case 'Fixed_Epoch':
            source_directories = (
                '2023-09-26 Rosenbrock NoLRClipping',
                '2023-09-24 UCI_Energy NoLRClipping',
                '2023-09-24 UCI_Protein NoLRClipping',
                '2023-09-24 Fashion-MNIST NoLRClipping',
                '2023-09-24 SVHN NoLRClipping',
                '2023-09-21 CIFAR-10 NoLRClipping',
            )
        case 'Fixed_Runtime_Training':
            source_directories = (
                '2023-09-27 UCI_Energy ASHA_Time_Training',
                '2023-09-27 UCI_Protein ASHA_Time_Training',
                '2023-09-27 Fashion-MNIST ASHA_Time_Training',
                '2023-09-27 SVHN ASHA_Time_Training',
                '2023-09-27 CIFAR-10 ASHA_Time_Training',
            )
        case 'Fixed_Runtime_Validation':
            source_directories = (
                '2023-09-27 UCI_Energy ASHA_Time_Validation',
                '2023-09-27 UCI_Protein ASHA_Time_Validation',
                '2023-09-27 Fashion-MNIST ASHA_Time_Validation',
                '2023-09-27 SVHN ASHA_Time_Validation',
                '2023-09-27 CIFAR-10 ASHA_Time_Validation',
            )
    with open(f'./plots/paper_ICML_NoClipping/Hyperparameters_{suite}.tex', 'w') as table:
        for directory_idx, directory in enumerate(source_directories):
            dataset_name = directory.split(' ')[1].replace('_', ' ')
            num_algorithms = len([d for d in os.scandir(f'/scratch/dir/ImprovingKFAC/{directory}')
                                  if plot.KEY_TO_LABEL.get(d.name, False)])
            table.write(r'\multirow{'+ str(num_algorithms) + r'}{*}' + f'{{{dataset_name}}}\n')
            for algorithm, label in plot.KEY_TO_LABEL.items():
                best_runs_path = os.path.join('/scratch/dir/ImprovingKFAC', directory, algorithm)
                if not label:
                    continue
                if not os.path.exists(best_runs_path):
                    continue
                if algorithm == 'AdamQLR_Damped_Fisher':
                    # Duplicate folder for ease of labelling ablation plots
                    continue
                config_file = os.path.join(next(os.scandir(best_runs_path)), 'config.json')
                with open(config_file, 'r') as config_raw:
                    config_data = json.load(config_raw)

                table.write('& ' + plot.KEY_TO_LABEL[algorithm] + ' \t& ')

                if dataset_name != 'Rosenbrock':
                    table.write(f"{config_data['batch_size']} \t& ")
                else: table.write("{---} \t& ")

                if algorithm in ('SGD', 'SGDmwd', 'Adam'):
                    table.write(f"{config_data['optimiser']['learning_rate']:.2e} \t& ")
                else: table.write("{---} \t& ")

                # if 'lr_clipping' in config_data['optimiser']:
                #     table.write(f"{config_data['optimiser']['lr_clipping']:.3f} \t& ")
                # else: table.write("{---} \t& ")

                if algorithm == 'SGDmwd':
                    table.write(f"{config_data['optimiser']['momentum']:.3f} \t& "
                                f"{config_data['optimiser']['add_decayed_weights']:.2e} \t& ")
                else: table.write("{---} \t& {---} \t& ")

                if 'Damped' in algorithm or algorithm in ('KFAC', 'AdamQLR_NoHPO'):
                    table.write(f"{config_data['optimiser']['initial_damping']:.2e} \t& ")
                else: table.write("{---} \t&")

                if config_data['optimiser'].get('damping_increase_factor', None):
                    table.write(f"{config_data['optimiser']['damping_decrease_factor']:.1f} \t& "
                                f"{config_data['optimiser']['damping_increase_factor']:.1f} ")
                else: table.write("{---} \t& {---} ")

                table.write(r'\\' + '\n')
            if directory_idx + 1 != len(source_directories):
                table.write(r'\midrule' + '\n')
        table.write(r'\bottomrule')


def ablation_plots():
    plot_configs = dict(
        Damping=dict(directories=('2023-09-24 Fashion-MNIST NoLRClipping',
                                  '2023-09-21 CIFAR-10 NoLRClipping'),
                     algorithms=('Adam',
                                 'AdamQLR_Undamped_NoLRClipping',
                                 'AdamQLR_Damped_NoLRClipping')),
        Curvature=dict(directories=('2023-09-24 Fashion-MNIST NoLRClipping',
                                    '2023-09-21 CIFAR-10 NoLRClipping',),
                       algorithms=('Adam',
                                   'AdamQLR_Damped_Hessian_NoLRClipping',
                                   'AdamQLR_Damped_Fisher_NoLRClipping')),
    )
    for plot_name, plot_config in plot_configs.items():
        for directory in plot_config['directories']:
            dataset_name = directory.split(' ')[1]
            for metric in ('Loss/Training', 'Loss/Test'):
                loss_name = metric[5:]
                with plot.inhibit_plt_show(), paper_theme():
                    axes = plot.plot_best_run_envelopes(
                        f'/scratch/dir/ImprovingKFAC/{directory}',
                        metric,
                        log_x_axis=False,
                        included_algorithms=plot_config['algorithms'],
                        aggregation='median')
                axes[0].set_ylabel(f'{loss_name} Loss')
                axes[0].margins(x=0)
                # match dataset_name:
                #     case 'Fashion-MNIST': plt.ylim(7e-2, 2.5e0)
                #     case 'CIFAR-10': plt.ylim(7e-1, 2e1)
                tight_savefig(f'./plots/paper_ICML_NoClipping/Ablation_{plot_name}_{dataset_name}_{loss_name}Loss_CleanAxesLimits.pdf')


def imagenet_plots():
    sgd_data = pd.read_csv('./runs/ImageNet/sgd_baseline.csv')
    adam_data = pd.read_csv('./runs/ImageNet/adam.csv')
    adamqlr_data = pd.read_csv('./runs/ImageNet/adam_qlr.csv')

    for metric in ('train_accuracy', 'test_accuracy'):
        match metric:
            case 'train_accuracy': accuracy_name = 'Training'
            case 'test_accuracy': accuracy_name = 'Test'
        with plot.inhibit_plt_show(), paper_theme():
            plt.plot(sgd_data['accumulated_submission_time'],
                     sgd_data[metric],
                     label="SGD-ImageNet",
                     **plot.KEY_TO_STYLE['SGDmwd'])
            plt.plot(adam_data['accumulated_submission_time'],
                     adam_data[metric],
                     label="Adam (Untuned)",
                     **plot.KEY_TO_STYLE['Adam'])
            plt.plot(adamqlr_data['accumulated_submission_time'],
                     adamqlr_data[metric],
                     label="AdamQLR (Untuned)",
                     **plot.KEY_TO_STYLE['AdamQLR_NoHPO'])
            plt.legend()
        plt.xlabel('Runtime (s)')
        plt.ylabel(f'{accuracy_name} Accuracy')
        plt.xlim(0, None)
        plt.ylim(0, None)
        tight_savefig(f'./plots/paper_NoLRClipping/ImageNet_{accuracy_name}Accuracy.pdf')


if __name__ == '__main__':
    rosenbrock_trajectory_plot()
    loss_evolution_plots()
    sensitivity_plots()
    hyperparameter_table('Fixed_Epoch')
    hyperparameter_table('Fixed_Runtime_Training')
    hyperparameter_table('Fixed_Runtime_Validation')
    ablation_plots()
    # imagenet_plots()
