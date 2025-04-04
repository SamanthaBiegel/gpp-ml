import torch
import numpy as np
import pandas as pd
import argparse
import datetime
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.model_selection import StratifiedKFold, train_test_split
from data.dataset import gpp_dataset, compute_train_stats, add_stratification_target
from models.lstm_model import Model as LSTMModel
from models.mlp_model import Model as MLPModel
from utils.utils import set_seed
from utils.train_model import train_model
from utils.train_test_loops import test_loop
from utils.evaluate_model import evaluate_model, compute_metrics
import torch.multiprocessing as mp

np.seterr(invalid='ignore')

def get_hyperparameter_space(model_type):
    hparam_space = {
        'hidden_dim': [32, 64, 128, 256, 512],
        'lr': [1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4],
        'scheduler_patience': [5, 10, 20, 30],
        'scheduler_factor': [0.1, 0.5, 0.9],
        'weight_decay': [0.01, 0.001, 0.0001, 0.00001, 0],
        'batch_size': [16, 32, 64, 128, 256]
    }
    
    if model_type == 'LSTM':
        hparam_space.update({
            'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'num_layers': [1, 2, 3, 4, 5],
        })
    
    return hparam_space

def sample_hyperparameters(hparam_space, model_type):
    hparams = {}
    for key, values in hparam_space.items():
        sampled_value = np.random.choice(values)
        if key in ['hidden_dim', 'scheduler_patience', 'num_layers', 'batch_size']:
            hparams[key] = int(sampled_value)
        else:
            hparams[key] = sampled_value
    
    if model_type == 'LSTM' and hparams.get('num_layers', 1) == 1:
        hparams['dropout'] = 0

    return hparams

def run_fold(rank, data, site_stratify, train_test_indices, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue, device):
    # Run the training and validation for a single fold

    has_categorical_features = True if len(categorical_features) > 0 else False

    # Prepare data
    train_val_index, test_index = train_test_indices
    train_val_sites = site_stratify.index.values[train_val_index]
    test_sites = site_stratify.index.values[test_index]
    site_stratify_train_val = site_stratify.loc[train_val_sites]
    train_sites, val_sites = train_test_split(site_stratify_train_val.index, stratify=site_stratify_train_val['stratify'], test_size=0.2)
    data_train = data.loc[train_sites]
    data_val = data.loc[val_sites]
    data_test = data.loc[test_sites]

    writer = None
    
    best_validation_score = np.inf
    best_hyperparameters = None

    # Create per-site stratification targets for inner CV
    site_stratify_cv = site_stratify.loc[train_sites]
    sites_cv = site_stratify_cv.index.values
    stratify_values_cv = site_stratify_cv['stratify'].values

    kf_cv = StratifiedKFold(n_splits=3, shuffle=True)
    cv_split = list(kf_cv.split(sites_cv, stratify_values_cv))

    train_ds_list = []
    val_ds_list = []
    # Create datasets for inner CV
    for train_index_cv, val_index_cv in cv_split:
        train_sites_cv = sites_cv[train_index_cv]
        val_sites_cv = sites_cv[val_index_cv]

        data_train_cv = data.loc[train_sites_cv]
        data_val_cv = data.loc[val_sites_cv]

        train_stats_cv = compute_train_stats(data_train_cv, numerical_features)
        train_ds_cv = gpp_dataset(data_train_cv, train_stats_cv, numerical_features, categorical_features, test=False)
        val_ds_cv = gpp_dataset(data_val_cv, train_stats_cv, numerical_features, categorical_features, test=True)

        train_ds_list.append(train_ds_cv)
        val_ds_list.append(val_ds_cv)

    # Hyperparameter tuning on inner CV loop
    for j in tqdm(range(args.num_trials), desc=f"Device {rank}"):
        hparams = sample_hyperparameters(hparam_space, args.model_type)

        total_cv_rmse = 0

        # Test hyperparameters with 3-fold cross-validation
        for k in range(3):
            train_ds_cv = train_ds_list[k]
            val_ds_cv = val_ds_list[k]

            train_dl_cv = DataLoader(train_ds_cv, batch_size=hparams['batch_size'], shuffle=True, num_workers=4)
            val_dl_cv = DataLoader(val_ds_cv, batch_size=1, shuffle=False, num_workers=4)

            if args.model_type == 'LSTM':
                model = LSTMModel(
                    input_dim=len(numerical_features), 
                    conditional_dim=conditional_dim, 
                    hidden_dim=hparams['hidden_dim'], 
                    dropout=hparams['dropout'], 
                    num_layers=hparams['num_layers'],
                    attention=args.attention,
                    layernorm=args.layernorm
                )
            elif args.model_type == 'MLP':
                model = MLPModel(
                    input_dim=len(numerical_features), 
                    conditional_dim=conditional_dim, 
                    hidden_dim=hparams['hidden_dim']
                )
            model = model.to(device=device)

            optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=hparams['scheduler_patience'], factor=hparams['scheduler_factor'])

            cv_fold_val_metrics, model = train_model(train_dl_cv, val_dl_cv, model, optimizer, scheduler, args.n_epochs, device, args.patience, writer, args.early_stopping, has_categorical_features)

            val_rmse_cv = cv_fold_val_metrics['rmse']
            total_cv_rmse += val_rmse_cv

        mean_val_rmse = total_cv_rmse / 3
        
        if mean_val_rmse < best_validation_score:
            best_validation_score = mean_val_rmse
            best_hyperparameters = hparams

    print(f"Best hyperparameters - fold {rank} - {best_hyperparameters}")

    # Train model again with best hyperparameters
    writer = SummaryWriter(log_dir=f"{base_path}/runs/{filename}/fold_{rank}")

    train_stats = compute_train_stats(data_train, numerical_features)
    train_ds = gpp_dataset(data_train, train_stats, numerical_features, categorical_features, test=False)
    val_ds = gpp_dataset(data_val, train_stats, numerical_features, categorical_features, test=True)
    train_dl = DataLoader(train_ds, batch_size=best_hyperparameters['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

    if args.model_type == 'LSTM':
        best_model = LSTMModel(
            input_dim=len(numerical_features), 
            conditional_dim=conditional_dim, 
            hidden_dim=best_hyperparameters['hidden_dim'], 
            dropout=best_hyperparameters['dropout'], 
            num_layers=best_hyperparameters['num_layers'],
            attention=args.attention,
            layernorm=args.layernorm
        )
    elif args.model_type == 'MLP':
        best_model = MLPModel(
            input_dim=len(numerical_features), 
            conditional_dim=conditional_dim, 
            hidden_dim=best_hyperparameters['hidden_dim']
        )
    best_model = best_model.to(device=device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_hyperparameters['lr'], weight_decay=best_hyperparameters['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=best_hyperparameters['scheduler_patience'], factor=best_hyperparameters['scheduler_factor'])

    best_val_metrics, best_model = train_model(train_dl, val_dl, best_model, optimizer, scheduler, args.n_epochs, device, args.patience, writer, args.early_stopping, has_categorical_features)

    print(f"{args.model_type} validation metrics - fold {rank} - R2: {best_val_metrics['r2']:.4f} | RMSE: {best_val_metrics['rmse']:.4f} | NMAE: {best_val_metrics['nmae']:.4f} | Abs Bias: {best_val_metrics['abs_bias']:.4f} | NSE: {best_val_metrics['nse']:.4f}")

    torch.save(best_model.state_dict(), f"{base_path}/weights/{filename}_fold_{rank}.pt")
    writer.close()

    test_ds = gpp_dataset(data_test, train_stats, numerical_features, categorical_features, test=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    _, y_pred = test_loop(test_dl, best_model, device, has_categorical_features)

    test_metrics, data_test_eval = evaluate_model(test_dl, y_pred)

    fold_df = data_test_eval[['TIMESTAMP', 'GPP_NT_VUT_REF', 'gpp_pred']]

    results_queue.put((fold_df, best_val_metrics, test_metrics, best_hyperparameters))

    print(f"{args.model_type} test metrics - fold {rank} - R2: {test_metrics['r2']:.4f} | RMSE: {test_metrics['rmse']:.4f} | NMAE: {test_metrics['nmae']:.4f} | Abs Bias: {test_metrics['abs_bias']:.4f} | NSE: {test_metrics['nse']:.4f}")
    print(f"Best hyperparameters - fold {rank} - {best_hyperparameters}")

def process_fold(rank, data, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue, folds, site_stratify):
    # Process a single fold in a separate process
    gpu_id = rank
    if args.device == 'cuda':
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)

    train_test_indices = folds[rank]

    run_fold(rank, data, site_stratify, train_test_indices, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue, device)

    if rank == 0:
        aggregate_results(results_queue, base_path, filename, args)

def aggregate_results(results_queue, base_path, filename, args):
    # Aggregate results from the queue
    all_dfs_out = []
    all_val_metrics_list = []
    all_test_metrics_list = []
    best_hyperparams = []

    while len(all_dfs_out) < 5:
        fold_df, fold_val_metrics, fold_test_metrics, fold_hyperparams = results_queue.get()
        all_dfs_out.append(fold_df)
        all_val_metrics_list.append(fold_val_metrics)
        all_test_metrics_list.append(fold_test_metrics)
        best_hyperparams.append(fold_hyperparams)

    # Aggregate and save the results
    all_val_metrics = {key: [metrics[key] for metrics in all_val_metrics_list] for key in all_val_metrics_list[0].keys()}
    all_test_metrics = {key: [metrics[key] for metrics in all_test_metrics_list] for key in all_test_metrics_list[0].keys()}
    df_out = pd.concat(all_dfs_out)
    preds_filename = f'{base_path}/preds/{filename}.csv'
    df_out.to_csv(preds_filename)
    print(f'Predictions saved to {preds_filename}')
    print(f"{args.model_type} - Mean val R2: {np.mean(all_val_metrics['r2']):.4f} | Mean RMSE: {np.mean(all_val_metrics['rmse']):.4f} | Mean NMAE: {np.mean(all_val_metrics['nmae']):.4f} | Mean Abs Bias: {np.mean(all_val_metrics['abs_bias']):.4f} | Mean NSE: {np.mean(all_val_metrics['nse']):.4f}")
    print(f"{args.model_type} - Mean test R2: {np.mean(all_test_metrics['r2']):.4f} | Mean RMSE: {np.mean(all_test_metrics['rmse']):.4f} | Mean NMAE: {np.mean(all_test_metrics['nmae']):.4f} | Mean Abs Bias: {np.mean(all_test_metrics['abs_bias']):.4f} | Mean NSE: {np.mean(all_test_metrics['nse']):.4f}")
    metrics_total = compute_metrics(df_out['GPP_NT_VUT_REF'], df_out['gpp_pred'])
    print(f"{args.model_type} - Total R2: {metrics_total['r2']:.4f} | Total RMSE: {metrics_total['rmse']:.4f} | Total NMAE: {metrics_total['nmae']:.4f} | Total Abs Bias: {metrics_total['abs_bias']:.4f} | Total NSE: {metrics_total['nse']:.4f}")

def main(args):
    data = pd.read_csv('../data/fdk_v342_ml.csv', index_col='sitename', parse_dates=['TIMESTAMP'])
    numerical_features = ['TA_F_MDS', 'TA_DAY_F_MDS', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'VPD_DAY_F_MDS', 'PA_F', 'P_F', 'WS_F', 'FPAR']
    if args.model_type == 'LSTM' and args.extra_features:
        categorical_features = []
        numerical_features += ['whc']
    elif args.model_type == 'MLP' and args.extra_features:
        numerical_features += ['wscal']
        categorical_features = []
    else:
        categorical_features = []
    print("Extra features:", args.extra_features, "Categorical features:", categorical_features)
    if len(categorical_features) > 0:
        data_cat = pd.get_dummies(data[categorical_features], columns=categorical_features, drop_first=True)
        conditional_dim = len(data_cat.columns)
        data = pd.concat([data, data_cat], axis=1)
    else:
        conditional_dim = 0

    # Create per-site stratification targets
    stratified_data = add_stratification_target(data)
    site_stratify = stratified_data[['stratify']]
    sites = site_stratify.index.values
    stratify_values = site_stratify['stratify'].values

    # Generate the folds
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    folds = list(kf.split(sites, stratify_values))
    
    base_path = "../models"
    filename = f"{args.model_type}_lfo_nested_tuning_{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}"
    runs_dir = os.path.join(base_path, 'runs', filename)
    weights_dir = os.path.join(base_path, 'weights')
    preds_dir = os.path.join(base_path, 'preds')
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    
    hparam_space = get_hyperparameter_space(args.model_type)
    results_queue = mp.Queue()
    device_count = torch.cuda.device_count()
    if device_count >= 5:
        # Process all folds in parallel if enough devices are available
        world_size = 5
        mp.spawn(process_fold, args=(data, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue, folds, site_stratify), nprocs=world_size, join=True)
    elif 0 < device_count < 5 or args.device == 'mps':
        # Process all folds sequentially on the first GPU or MPS
        if args.device == 'cuda' and device_count >=1:
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            print("Processing folds sequentially on cuda:0")
        elif args.device == 'mps':
            device = torch.device('mps')
            print("Processing folds sequentially on mps device")
        else:
            device = torch.device('cpu')
            print("Processing folds sequentially on CPU")

        for fold_idx in tqdm(range(4, -1, -1), desc="Sequential fold processing"):
            train_test_indices = folds[fold_idx]
            print(f"Processing fold {fold_idx} on device {device}")
            run_fold(
                rank=fold_idx,
                data=data,
                site_stratify=site_stratify,
                train_test_indices=train_test_indices,
                base_path=base_path,
                filename=filename,
                args=args,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                conditional_dim=conditional_dim,
                hparam_space=hparam_space,
                results_queue=results_queue,
                device=device
            )
        aggregate_results(results_queue, base_path, filename, args)
    else:   
        print("Not enough devices available for multiprocessing")

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', '--device', default='cuda', type=str, help='Indices of GPU to enable')
    parser.add_argument('-model', '--model_type', default='LSTM', type=str, choices=['LSTM', 'MLP'], help='Model type to use')
    parser.add_argument('-a', '--attention', action=argparse.BooleanOptionalAction, help='Whether to use attention mechanism in LSTM model')
    parser.add_argument('-ln', '--layernorm', action=argparse.BooleanOptionalAction, help='Whether to use layer normalization in LSTM model')
    parser.add_argument('-e', '--n_epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('-es', '--early_stopping', action=argparse.BooleanOptionalAction, default=True, help='Whether to use early stopping')
    parser.add_argument('-p', '--patience', default=10, type=int, help='Number of iterations (patience threshold) used for early stopping')
    parser.add_argument('-t', '--num_trials', default=20, type=int, help='Number of trials for hyperparameter tuning')
    parser.add_argument('-c', '--extra_features', action=argparse.BooleanOptionalAction, help='Whether to include extra features')
    parser.add_argument('-s', '--seed', default=31, type=int, help='Seed for reproducibility')
    args = parser.parse_args()

    print("Starting leave-fold-out training and validation on model:")
    print(f"> Device: {args.device}")
    print(f"> Model type: {args.model_type}")
    if args.model_type == 'LSTM':
        print(f"> Attention mechanism: {args.attention}")
        print(f"> Layer normalization: {args.layernorm}")
    print(f"> Epochs: {args.n_epochs}")
    if args.early_stopping:
        print(f"> Early stopping after {args.patience} epochs without improvement")
    print(f"> Number of trials: {args.num_trials}")
    print(f"> Extra features: {args.extra_features}")

    print(f"> Number of available devices: {torch.cuda.device_count()}")

    set_seed(args.seed)
    print(f"Seed set to {args.seed}")

    main(args)