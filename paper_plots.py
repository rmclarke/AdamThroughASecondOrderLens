import json
import os
from contextlib import contextmanager

import matplotlib.pyplot as plt

import plot


CORE_ALGORITHMS = ('SGD',
                   'SGDmwd',
                   'Adam',
                   'AdamQLR_Damped',
                   'AdamQLR_NoHPO',
                   'KFAC')
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
            '/scratch/dir/ImprovingKFAC/2023-09-26 Rosenbrock ANLRClipping',
            ['GD', 'GDmwd', 'Adam', 'AdamQLR_Damped_Hessian', 'AdamQLR_NoHPO'])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    tight_savefig('./plots/paper_ANLRClipping/RosenbrockTrajectory.pdf')


def loss_evolution_plots():
    source_directories = (
        '2023-09-24 UCI_Energy ANLRClipping',
        '2023-09-24 UCI_Protein ANLRClipping',
        '2023-09-24 Fashion-MNIST ANLRClipping',
        '2023-09-21 CIFAR-10 ANLRClipping',
        '2023-09-24 SVHN ANLRClipping',
        '2023-09-27 UCI_Energy ASHA_Time_Training',
        '2023-09-27 UCI_Protein ASHA_Time_Training',
        '2023-09-27 Fashion-MNIST ASHA_Time_Training',
        '2023-09-27 CIFAR-10 ASHA_Time_Training',
        '2023-09-27 SVHN ASHA_Time_Training',
        '2023-09-27 UCI_Energy ASHA_Time_Validation',
        '2023-09-27 UCI_Protein ASHA_Time_Validation',
        '2023-09-27 Fashion-MNIST ASHA_Time_Validation',
        '2023-09-27 CIFAR-10 ASHA_Time_Validation',
        '2023-09-27 SVHN ASHA_Time_Validation',
    )
    for directory in source_directories:
        dataset_name = directory.split(' ')[1]
        match dataset_name:
            case 'UCI_Energy': break_point=300
            case 'UCI_Protein': break_point=300
            case 'Fashion-MNIST': break_point = 30
            case 'SVHN': break_point=250
            case 'CIFAR-10': break_point=2000
            case 'PennTreebank_GPT2_Reset': break_point=4000
        valid_metrics = ['Loss/Training',
                         'Loss/Test',
                         'Adaptive/Learning_Rate',]
        if not dataset_name.startswith('UCI'):
            valid_metrics.extend(['Accuracy/Training',
                                  'Accuracy/Test'])
        if 'ASHA_Time_' in directory:
            dataset_name += directory.split(' ')[-1]
        for metric in valid_metrics:
            with plot.inhibit_plt_show(), paper_theme():
                axes = plot.plot_best_run_envelopes(
                    f'/scratch/dir/ImprovingKFAC/{directory}',
                    metric,
                    log_x_axis=False,
                    included_algorithms=CORE_ALGORITHMS,
                    aggregation='median',
                    break_x_axis=break_point)
                if metric.startswith('Accuracy'):
                    axes[0].set_yscale('linear')
                    axes[0].set_ylim(0, 1.0)
                if (dataset_name.startswith('Fashion-MNISTASHA_Time_')
                    and metric.startswith('Loss/')):
                    # 2e-6
                    axes[0].set_ylim(None, 7)
            metric_name = metric.split('/')[1]
            if metric.startswith('Loss'):
                suffix = ' Loss'
            elif metric.startswith('Accuracy'):
                suffix = ' Accuracy'
            else:
                suffix = ''
            pretty_metric_name = ' '.join(metric_name.split('_'))
            axes[0].set_ylabel(f'{pretty_metric_name}{suffix}')
            tight_savefig(f'./plots/paper_ANLRClipping/{dataset_name}_{metric_name}{suffix.split(" ")[-1]}.pdf')


def sensitivity_plots():
    directories = (
        '/scratch2/dir/ImprovingKFAC/ray/2023-09-24T09:38:35.097408__fashion_mnist__AdamQLR_Damped__ASHA Sensitivity_Amplification',
        '/scratch2/dir/ImprovingKFAC/ray/2023-09-24T09:38:35.097408__fashion_mnist__AdamQLR_Damped__ASHA Sensitivity_BatchSize',
        '/scratch2/dir/ImprovingKFAC/ray/2023-09-24T09:38:35.097408__fashion_mnist__AdamQLR_Damped__ASHA Sensitivity_InitialDamping',
        '/scratch2/dir/ImprovingKFAC/ray/2023-09-24T09:38:35.097408__fashion_mnist__AdamQLR_Damped__ASHA Sensitivity_LRClipping',
        '/scratch2/dir/ImprovingKFAC/ray/2023-09-24T09:38:35.097408__fashion_mnist__AdamQLR_Damped__ASHA Sensitivity_SteppingFactor',
    )
    for directory in directories:
        ablation_type = directory.split('_')[-1]
        for metric in ('Loss/Training', 'Loss/Test'):
            with (plot.inhibit_plt_show(), paper_theme()):
                axes = plot.plot_ablation_trends(directory,
                                                 metric,
                                                 log_x_axis=False,
                                                 aggregation='median',
                                                 break_x_axis=14 if ablation_type == 'BatchSize' else False)
            loss_name = metric[5:]
            axes[0].set_ylabel(f'{loss_name} Loss')
            if ablation_type != 'BatchSize':
                plt.xlim(0, 14)
            tight_savefig(f'./plots/paper_ANLRClipping/Sensitivity_{ablation_type}_{loss_name}Loss.pdf')

    for metric in ('Loss/Training', 'Loss/Test'):
        with plot.inhibit_plt_show(), paper_theme():
            axes = plot.plot_best_run_envelopes(
                '/scratch/dir/ImprovingKFAC/2023-09-24 Fashion-MNIST ANLRClipping',
                metric,
                log_x_axis=False,
                included_algorithms=CORE_ALGORITHMS,
                aggregation='median',
                break_x_axis=14)
        loss_name = metric[5:]
        axes[0].set_ylabel(f'{loss_name} Loss')
        tight_savefig(f'./plots/paper_ANLRClipping/Sensitivity_ReprisedFashion-MNIST_{loss_name}Loss.pdf')


def hyperparameter_table():
    source_directories = (
        '2023-09-26 Rosenbrock ANLRClipping',
        '2023-09-24 UCI_Energy ANLRClipping',
        '2023-09-24 UCI_Protein ANLRClipping',
        '2023-09-24 Fashion-MNIST ANLRClipping',
        '2023-09-21 CIFAR-10 ANLRClipping',
        '2023-09-24 SVHN ANLRClipping',
    )
    with open('./plots/paper_ANLRClipping/Hyperparameters.tex', 'w') as table:
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
                    table.write(f"{config_data['optimiser']['learning_rate']:.4e} \t& ")
                else: table.write("{---} \t& ")

                if 'lr_clipping' in config_data['optimiser']:
                    table.write(f"{config_data['optimiser']['lr_clipping']:.3f} \t& ")
                else: table.write("{---} \t& ")

                if algorithm == 'SGDmwd':
                    table.write(f"{config_data['optimiser']['momentum']:.4f} \t& "
                                f"{config_data['optimiser']['add_decayed_weights']:.4e} \t& ")
                else: table.write("{---} \t& {---} \t& ")

                if 'Damped' in algorithm or algorithm in ('KFAC', 'AdamQLR_NoHPO'):
                    table.write(f"{config_data['optimiser']['initial_damping']:.4e} \t& ")
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
        Damping=dict(directories=('2023-09-24 Fashion-MNIST ANLRClipping',
                                  '2023-09-21 CIFAR-10 ANLRClipping'),
                     algorithms=('Adam',
                                 'AdamQLR_Undamped',
                                 'AdamQLR_Damped')),
        Curvature=dict(directories=('2023-09-24 Fashion-MNIST ANLRClipping',
                                    '2023-09-21 CIFAR-10 ANLRClipping',),
                       algorithms=('Adam',
                                   'AdamQLR_Damped_Hessian',
                                   'AdamQLR_Damped_Fisher')),
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
                # match dataset_name:
                #     case 'Fashion-MNIST': plt.ylim(7e-2, 2.5e0)
                #     case 'CIFAR-10': plt.ylim(7e-1, 2e1)
                tight_savefig(f'./plots/paper_ANLRClipping/Ablation_{plot_name}_{dataset_name}_{loss_name}Loss.pdf')


if __name__ == '__main__':
    rosenbrock_trajectory_plot()
    loss_evolution_plots()
    sensitivity_plots()
    hyperparameter_table()
    ablation_plots()
