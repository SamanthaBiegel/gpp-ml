import torch
import numpy as np
import pandas as pd
import argparse
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data.dataset import gpp_dataset, compute_train_stats
from models.lstm_model import Model as LSTMModel
from models.mlp_model import Model as MLPModel
from utils.utils import set_seed
from utils.train_model import train_model
from utils.train_test_loops import test_loop
from utils.evaluate_model import evaluate_model
import torch.multiprocessing as mp
import random

np.seterr(invalid='ignore')

def get_hyperparameter_space(model_type):
    hparam_space = {
        'hidden_dim': [4, 8, 16, 32, 64],
        'lr': [1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4],
        'scheduler_patience': [5, 10, 15],
        'scheduler_factor': [0.1, 0.5, 0.9],
        'weight_decay': [0.01, 0.001, 0.0001, 0.00001, 0],
        'batch_size': [4, 8, 16, 32]
    }
    
    if model_type == 'LSTM':
        hparam_space.update({
            'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'num_layers': [1, 2, 3],
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

def generate_val_periods(site_data, test_period_start, test_period_end):
    month_start = test_period_start.month
    min_date = site_data['TIMESTAMP'].min()
    max_date = site_data['TIMESTAMP'].max()
    val_periods = []
    current_start_year = min_date.year
    while True:
        current_start_date = pd.Timestamp(year=current_start_year, month=month_start, day=1)
        current_end_date = current_start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        if current_end_date > max_date:
            break
        if current_start_date == test_period_start:
            current_start_year += 1
            continue
        val_periods.append((current_start_date, current_end_date))
        current_start_year += 1
    return val_periods


def run_site_period(rank, site, test_period_start, test_period_end, site_data, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue):
    # Run the model for a specific site and period

    gpu_id = rank
    if args.device == 'cuda':
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)

    cat = True if len(categorical_features) > 0 else False

    train_val_data = site_data[(site_data['TIMESTAMP'] < test_period_start) | (site_data['TIMESTAMP'] > test_period_end)]
    test_data = site_data[(site_data['TIMESTAMP'] >= test_period_start) & (site_data['TIMESTAMP'] <= test_period_end)]

    best_validation_score = np.inf
    best_hyperparameters = None

    val_years = generate_val_periods(site_data, test_period_start, test_period_end)

    # Pick 5 random years for validation
    if len(val_years) < 5:
        subset_val_years = val_years
    else:
        subset_val_years = random.sample(val_years, 5)

    # Inner cross-validation for hyperparameter tuning
    for j in tqdm(range(args.num_trials), desc=f"Device {gpu_id}"):
        hparams = sample_hyperparameters(hparam_space, args.model_type)

        total_cv_rmse = 0
        for val_period_start, val_period_end in subset_val_years:
            train_data_cv = train_val_data[(train_val_data['TIMESTAMP'] < val_period_start) | (train_val_data['TIMESTAMP'] > val_period_end)]
            val_data_cv = train_val_data[(train_val_data['TIMESTAMP'] >= val_period_start) & (train_val_data['TIMESTAMP'] <= val_period_end)]

            train_stats_cv = compute_train_stats(train_data_cv, numerical_features)
            train_ds_cv = gpp_dataset(train_data_cv, train_stats_cv, numerical_features, categorical_features, test=False)
            val_ds_cv = gpp_dataset(val_data_cv, train_stats_cv, numerical_features, categorical_features, test=True)

            train_dl_cv = DataLoader(train_ds_cv, batch_size=hparams['batch_size'], shuffle=True, num_workers=0)
            val_dl_cv = DataLoader(val_ds_cv, batch_size=1, shuffle=False, num_workers=0)

            if args.model_type == 'LSTM':
                model = LSTMModel(
                    input_dim=len(numerical_features),
                    conditional_dim=conditional_dim, 
                    hidden_dim=hparams['hidden_dim'], 
                    dropout=hparams['dropout'], 
                    num_layers=hparams['num_layers'],
                    attention=args.attention
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

            cv_fold_val_metrics, model = train_model(train_dl_cv, val_dl_cv, model, optimizer, scheduler, args.n_epochs, device, args.patience, None, args.early_stopping, cat)

            val_rmse_cv = cv_fold_val_metrics['rmse']
            total_cv_rmse += val_rmse_cv

        mean_val_rmse = total_cv_rmse / len(subset_val_years)
        if mean_val_rmse < best_validation_score:
            best_validation_score = mean_val_rmse
            best_hyperparameters = hparams

    print(f"Best hyperparameters for site {site} - period {test_period_start.strftime('%Y%m%d')}: {best_hyperparameters}")

    # Train model again with best hyperparameters on all training data
    test_period_str = test_period_start.strftime('%Y%m%d')
    writer = SummaryWriter(log_dir=f"{base_path}/runs/{filename}/site_{site}_period_{test_period_str}")

    val_year = random.choice(val_years)
    final_val = train_val_data[(train_val_data['TIMESTAMP'] >= val_year[0]) & (train_val_data['TIMESTAMP'] <= val_year[1])]
    final_train = train_val_data[(train_val_data['TIMESTAMP'] < val_year[0]) | (train_val_data['TIMESTAMP'] > val_year[1])]
    train_stats = compute_train_stats(final_train, numerical_features)
    train_ds = gpp_dataset(final_train, train_stats, numerical_features, categorical_features, test=False)
    val_ds = gpp_dataset(final_val, train_stats, numerical_features, categorical_features, test=True)
    train_dl = DataLoader(train_ds, batch_size=best_hyperparameters['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
    if args.model_type == 'LSTM':
                model = LSTMModel( 
                    input_dim=len(numerical_features), 
                    conditional_dim=conditional_dim, 
                    hidden_dim=best_hyperparameters['hidden_dim'], 
                    dropout=best_hyperparameters['dropout'], 
                    num_layers=best_hyperparameters['num_layers'],
                    attention=args.attention,
                    layernorm=args.layernorm
                )
    elif args.model_type == 'MLP':
        model = MLPModel(
            input_dim=len(numerical_features), 
            conditional_dim=conditional_dim, 
            hidden_dim=best_hyperparameters['hidden_dim']
        )
    best_model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparameters['lr'], weight_decay=best_hyperparameters['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=best_hyperparameters['scheduler_patience'], factor=best_hyperparameters['scheduler_factor'])

    best_val_metrics, best_model = train_model(train_dl, val_dl, best_model, optimizer, scheduler, args.n_epochs, device, args.patience, writer, args.early_stopping, cat)

    print(f"{args.model_type} validation metrics - site {site} - period {test_period_str} - R2: {best_val_metrics['r2']:.4f} | RMSE: {best_val_metrics['rmse']:.4f} | NMAE: {best_val_metrics['nmae']:.4f} | Abs Bias: {best_val_metrics['abs_bias']:.4f} | NSE: {best_val_metrics['nse']:.4f}")

    torch.save(best_model.state_dict(), f"{base_path}/weights/{filename}_site_{site}_period_{test_period_str}.pt")
    writer.close()

    test_ds = gpp_dataset(test_data, train_stats, numerical_features, categorical_features, test=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    _, y_pred = test_loop(test_dl, best_model, device, cat)
    test_metrics, data_test_eval = evaluate_model(test_dl, y_pred)

    fold_df = data_test_eval[['TIMESTAMP', 'GPP_NT_VUT_REF', 'gpp_pred']]

    results_queue.put((fold_df, best_val_metrics, test_metrics, best_hyperparameters))

    print(f"{args.model_type} test metrics - site {site} - period {test_period_str} - R2: {test_metrics['r2']:.4f} | RMSE: {test_metrics['rmse']:.4f} | NMAE: {test_metrics['nmae']:.4f} | Abs Bias: {test_metrics['abs_bias']:.4f} | NSE: {test_metrics['nse']:.4f}")
    print(f"Best hyperparameters - site {site} - period {test_period_str} - {best_hyperparameters}")

def main(rank, world_size, data, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue):
    gpu_id = rank
    if args.device == 'cuda':
        torch.cuda.set_device(gpu_id)

    # Determine starting months for each site using P_F and TA_F_MDS over all data for a site
    site_start_months = {}
    for site, group in data.groupby('sitename'):
        koeppen_code = group['koeppen_code'].iloc[0]
        group['month'] = group['TIMESTAMP'].dt.month

        if koeppen_code.startswith(('C', 'D', 'E', '-')):
            # Compute coldest month using TA_F_MDS
            monthly_avg_temp = group.groupby('month')['TA_F_MDS'].mean()
            coldest_month = monthly_avg_temp.idxmin()
            site_start_months[site] = coldest_month
        elif koeppen_code.startswith(('A', 'B')):
            # Compute wettest month using P_F
            monthly_mean_precip = group.groupby('month')['P_F'].mean()
            wettest_month = monthly_mean_precip.idxmax()
            site_start_months[site] = wettest_month

    # Generate test periods for each site (each year separately)
    test_periods = []
    for site, group in data.groupby('sitename'):
        starting_month = site_start_months[site]
        # Get min and max dates
        min_date = group['TIMESTAMP'].min()
        max_date = group['TIMESTAMP'].max()

        # Adjust start_year to the first year where we have data for the starting month
        if min_date.month > starting_month:
            start_year = min_date.year
        else:
            start_year = min_date.year

        # Generate test periods of full years starting from the determined month
        current_start_year = start_year
        while True:
            current_start_date = pd.Timestamp(year=current_start_year, month=starting_month, day=1)
            current_end_date = current_start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
            # Check if we have data for the entire period
            if current_end_date > max_date:
                break  # No more full years of data
            test_periods.append((site, current_start_date, current_end_date))
            current_start_year += 1

    groups = test_periods

    # Distribute the groups across available GPUs
    grouped_by_device = {i: [] for i in range(world_size)}
    for i, group in enumerate(groups):
        grouped_by_device[i % world_size].append(group)
    
    print(f"Number of groups per device: {[(i, len(grouped_by_device[i])) for i in range(world_size)]}")

    for group in tqdm(grouped_by_device[rank]):
        site, test_period_start, test_period_end = group
        site_data = data.loc[site]
        run_site_period(rank, site, test_period_start, test_period_end, site_data, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue)

    if rank == 0:
        all_dfs_out = []
        all_val_metrics_list = []
        all_test_metrics_list = []
        best_hyperparams = []

        while len(all_dfs_out) < len(groups):
            fold_df, fold_val_metrics, fold_test_metrics, fold_hyperparams = results_queue.get()
            all_dfs_out.append(fold_df)
            all_val_metrics_list.append(fold_val_metrics)
            all_test_metrics_list.append(fold_test_metrics)
            best_hyperparams.append(fold_hyperparams)

        all_val_metrics = {key: [metrics[key] for metrics in all_val_metrics_list] for key in all_val_metrics_list[0].keys()}
        all_test_metrics = {key: [metrics[key] for metrics in all_test_metrics_list] for key in all_test_metrics_list[0].keys()}
        df_out = pd.concat(all_dfs_out)
        print(df_out.index.nunique())
        preds_filename = f'{base_path}/preds/{filename}.csv'
        df_out.to_csv(preds_filename)
        print(f'Predictions saved to {preds_filename}')
        print(f"{args.model_type} - Mean val R2: {np.mean(all_val_metrics['r2']):.4f} | Mean RMSE: {np.mean(all_val_metrics['rmse']):.4f} | Mean NMAE: {np.mean(all_val_metrics['nmae']):.4f} | Mean Abs Bias: {np.mean(all_val_metrics['abs_bias']):.4f} | Mean NSE: {np.mean(all_val_metrics['nse']):.4f}")
        print(f"{args.model_type} - Mean test R2: {np.mean(all_test_metrics['r2']):.4f} | Mean RMSE: {np.mean(all_test_metrics['rmse']):.4f} | Mean NMAE: {np.mean(all_test_metrics['nmae']):.4f} | Mean Abs Bias: {np.mean(all_test_metrics['abs_bias']):.4f} | Mean NSE: {np.mean(all_test_metrics['nse']):.4f}")
          
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', '--device', default='cuda', type=str, help='Indices of GPU to enable')
    parser.add_argument('-model', '--model_type', default='LSTM', type=str, choices=['LSTM', 'MLP'], help='Model type to use')
    parser.add_argument('-a', '--attention', action=argparse.BooleanOptionalAction, help='Whether to use attention mechanism in LSTM model')
    parser.add_argument('-ln', '--layernorm', action=argparse.BooleanOptionalAction, help='Whether to use layer normalization in LSTM model')
    parser.add_argument('-e', '--n_epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('-es', '--early_stopping', action=argparse.BooleanOptionalAction, help='Whether to use early stopping')
    parser.add_argument('-p', '--patience', default=10, type=int, help='Number of iterations (patience threshold) used for early stopping')
    parser.add_argument('-t', '--num_trials', default=40 , type=int, help='Number of trials for hyperparameter tuning')
    parser.add_argument('-s', '--seed', default=29, type=int, help='Seed for reproducibility')
    args = parser.parse_args()

    print("Starting site-period-level training and validation on model:")
    print(f"> Device: {args.device}")
    print(f"> Model type: {args.model_type}")
    if args.model_type == 'LSTM':
        print(f"> Attention mechanism: {args.attention}")
        print(f"> Layer normalization: {args.layernorm}")
    print(f"> Epochs: {args.n_epochs}")
    if args.early_stopping:
        print(f"> Early stopping after {args.patience} epochs without improvement")
    print(f"> Number of trials: {args.num_trials}")

    print(f"> Number of available devices: {torch.cuda.device_count()}")

    set_seed(args.seed)

    data = pd.read_csv('../data/processed/fdk_v342_ml.csv', parse_dates=['TIMESTAMP'])
    numerical_features = ['TA_F_MDS', 'TA_DAY_F_MDS', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'VPD_DAY_F_MDS', 'PA_F', 'P_F', 'WS_F', 'FPAR']
    categorical_features = []
    if len(categorical_features) > 0:
        data_cat = pd.get_dummies(data[categorical_features], columns=categorical_features, drop_first=True)
        conditional_dim = len(data_cat.columns)
        data = pd.concat([data, data_cat], axis=1)
    else:
        conditional_dim = 0

    print(f"Number of sites: {len(data.index.unique())}")
    base_path = "../models"
    filename = f"{args.model_type}_site_period_eval_{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}"

    hparam_space = get_hyperparameter_space(args.model_type)
    results_queue = mp.Queue()
    device_count = torch.cuda.device_count()
    if device_count >= 5:
        world_size = 5
        mp.spawn(main, args=(world_size, data, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue), nprocs=world_size, join=True)
    elif (1 <= device_count <= 5) or (args.device == 'mps'):
        world_size = 1
        rank = 0
        main(rank, world_size, data, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue)
    else:   
        print("Not enough devices available for multiprocessing")