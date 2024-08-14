import os
import sys
import pandas as pd
import csv

# Add the project's files to the python path
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
#file_path = os.path.dirname(os.path.abspath(''))  # for .ipynb notebook
sys.path.append(file_path)

import hydra
from src.utils import init_config, compute_panoptic_metrics, \
    grid_search_panoptic_partition, oracle_superpoint_clustering, \
    grid_search_panoptic_partition_FORinstance, panoptic_partition_FORinstance, \
    bayesian_optimize_panoptic_partition_FORinstance
import torch
from src.transforms import *
from src.utils.widgets import *
from src.data import *

# Very ugly fix to ignore lightning's warning messages about the
# trainer and modules not being connected
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' 

def add_results_to_csv(all_results_file_path, area, exp_tag, result_entry):
    # Extract results and combination dictionaries
    results = result_entry['results']
    combination = result_entry['combination']

    # Prepare the row to be added
    row = {'area': area, 'exp_tag': exp_tag}
    row.update(results)
    row.update(combination)

    # Check if the file exists to determine if headers need to be written
    file_exists = os.path.isfile(all_results_file_path)
    
    with open(all_results_file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=row.keys())

        # Write headers only if the file did not exist before
        if not file_exists:
            writer.writeheader()

        # Write the row to the CSV file
        writer.writerow(row)

# Function to perform optimization for a given experiment and checkpoint
def perform_optimization(experiment, ckpt_path, stage='train', i_cloud=0):
    # Parse the configs using hydra
    cfg = init_config(overrides=[
        f"experiment=panoptic/{experiment}",
        f"ckpt_path={ckpt_path}"
    ])

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup()

    # Load validation dataset
    if stage == 'val':
        dataset = datamodule.val_dataset
    elif stage == 'train':
        dataset = datamodule.train_dataset
    elif stage == 'test':
        dataset = datamodule.test_dataset
    else:
        raise ValueError("Invalid stage")
        
    # dataset.print_classes()

    model = hydra.utils.instantiate(cfg.model)
    if ckpt_path is not None:
        model = model._load_from_checkpoint(cfg.ckpt_path)

    model = model.eval().to(device)

    # Perform grid search optimization
    best_output, best_partitions, best_results, best_combination = grid_search_panoptic_partition_FORinstance(
        model,
        dataset,
        i_cloud=i_cloud,
        graph_kwargs=dict(
            k_max=k_max,
            radius=radius),
        partition_kwargs=dict(
            regularization=regularization,
            x_weight=x_weight,
            cutoff=cutoff),
        mode='pas',
        verbose=False)

    return best_output, best_partitions, best_results, best_combination


if __name__ == '__main__':
    version = 'nibio'

    experiment_tags = ['best'
                        ] # list of experiments / models to test
    
    area = 'NIBIO' # Each area must have a corresponding config file in experiment/panoptic.
    i_cloud = 0
    dataset_name = 'nibio'

    k_max = [10, 20]
    radius = [3, 5, 10]
    regularization = [10, 20]
    x_weight = [1e-3, 1e-1, 1]
    cutoff = [10000, 1000, 100]

    # For testing:
    #k_max = [10]
    #radius = [3, 5]
    #regularization = [10, 20]
    #x_weight = [1e-1]
    #cutoff = [1000]

    split = 'train'  # do not use 'test' split
    base_ckpt_file_path = '/home/vaschel/data/projects/superpoint_transformer_vschelbi/pretrained_models/'  # ckpt paths are: base_ckpt_file_path + experiment_tags[i] + '.ckpt'
    best_results_overall = {}     # Keep track of the best hyperparam combination, exp_tag, and results for each area
    all_results = {}
    all_results_file_path = f'/home/vaschel/data/projects/superpoint_transformer_vschelbi/logs/panoptic_parametrization/grid_search_{version}.csv'
    # max printing length
    max_len = 15

    combination_keys = ['k_max', 'radius', 'regulari.', 'x_weight', 'cutoff']

    print("PANOPTIC PARAMETRIZATION OPTIMIZATION")
    print("EXPERIMENTS:")
    print(experiment_tags)
    print()
    print("AREAS:")
    print(area)
    print(i_cloud)
    print()
    print("PARAMETER SPACE:")
    print(f"k_max: {k_max}")
    print(f"radius: {radius}")
    print(f"regularization: {regularization}")
    print(f"x_weight: {x_weight}")
    print(f"cutoff: {cutoff}")
    print()
    print("starting panoptic parametrization optimization...")

    for exp_tag in experiment_tags:
        ckpt_path = base_ckpt_file_path + exp_tag + '.ckpt'
        print()
        print(f"EXPERIMENT: {exp_tag}, AREA: {area}")
        experiment = dataset_name + '_' + exp_tag
        output, partitions, results, combination_list = perform_optimization(experiment, ckpt_path, split, i_cloud)
        
        combination = dict(zip(combination_keys, combination_list))

        if area not in all_results:
            all_results[area] = {}
        
        if exp_tag not in all_results[area]:
            all_results[area][exp_tag] = {}

        # keep all results and save in a csv file
        all_results[area][exp_tag] = {
            'results': results,
            'combination': combination,
            'model': exp_tag
        }
        add_results_to_csv(all_results_file_path, area, exp_tag, all_results[area][exp_tag])

        # keep best results if better tree f1 score is found
        if area not in best_results_overall or results['tree_f1'] > best_results_overall[area]['best_results']['tree_f1']:
            best_results_overall[area] = {
                'best_results': results,
                'best_combination': combination,
                'best_model': exp_tag
            }

    # output an intermediate result for each experiment
    print()
    print(f"Results for experiment {exp_tag}:")
    print(f"AREA: {area}")
    print(f"Current best Tree F1: {best_results_overall[area]['best_results']['tree_f1']}")
    print(f"Best combination: {best_results_overall[area]['best_combination']}")
    print(f"Best model: {best_results_overall[area]['best_model']}")
    print()
    

    # output the final best results
    print("-----------------------------------------------------------------")
    print("FINAL BEST RESULTS:")
    print()
    print(f"Area: {area}")
    print("Best panoptic setup: ")
    print(f"Tree F1={100 * best_results_overall[area]['best_results']['tree_f1']:0.2f}")
    with pd.option_context('display.precision', 2):
        print(pd.DataFrame(
            data=[best_results_overall[area]['best_combination'].values()],
            columns=[
                x[:max_len - 1] + '.' if len(x) > max_len else x
                for x in best_results_overall[area]['best_combination'].keys()
            ]
        ))
    print()
    # Print per-class results
    res = best_results_overall[area]['best_results']
    per_class_data = torch.column_stack([
        res['pq_per_class'].mul(100),
        res['precision_per_class'].mul(100),
        res['recall_per_class'].mul(100),
        (2 * (res['precision_per_class'] * res['recall_per_class']) / (res['precision_per_class'] + res['recall_per_class'] + 1e-8)).mul(100),
        res['tp_per_class'],
        res['fp_per_class'],
        res['fn_per_class']
    ])
    with pd.option_context('display.precision', 2):
        print(pd.DataFrame(
            data=per_class_data,
            index=['Ground and low vegetation', 'Tree'],
            columns=['PQ', 'PREC.', 'REC.', 'F1', 'TP', 'FP', 'FN']
        ))
    print()
    print("Panoptic parametrization optimization finished.")