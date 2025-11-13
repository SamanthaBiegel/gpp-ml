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
from utils.utils import set_seed
from utils.train_model import train_model
from utils.train_test_loops import test_loop
from utils.evaluate_model import evaluate_model, compute_metrics
import torch.multiprocessing as mp
import importlib

def make_model(kind, input_dim, conditional_dim, hparams, args):
    if kind == "LSTM":
        Model = importlib.import_module("models.lstm_model").Model
        return Model(input_dim=input_dim, conditional_dim=conditional_dim,
                     hidden_dim=hparams["hidden_dim"], dropout=hparams["dropout"],
                     num_layers=hparams["num_layers"], attention=args.attention, layernorm=args.layernorm)
    if kind == "MLP":
        Model = importlib.import_module("models.mlp_model").Model
        return Model(input_dim=input_dim, conditional_dim=conditional_dim,
                     hidden_dim=hparams["hidden_dim"])
    if kind == "TCN":
        Model = importlib.import_module("models.tcn_model").Model
        return Model(input_dim=input_dim, hidden_dim=hparams["hidden_dim"],
                     dropout=hparams["dropout"])
    raise ValueError(kind)

np.seterr(invalid='ignore')

def get_hyperparameter_space(model_type):
    hparam_space = {
        'hidden_dim': [64, 128, 256, 512],
        'lr': [1e-2, 5e-3, 1e-3, 5e-4],
        'scheduler_patience': [2, 3, 5],
        'scheduler_factor': [0.1, 0.5],
        'weight_decay': [1e-3, 1e-4, 1e-5, 0],
        'batch_size': [64, 128, 256]
    }
    
    if (model_type == 'LSTM' or model_type == 'TCN'):
        hparam_space.update({
            'dropout': [0, 0.1, 0.2, 0.3, 0.4],
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

def run_fold(rank, data, site_stratify, train_test_indices, base_path, filename, args,
             numerical_features, categorical_features, conditional_dim,
             hparam_space, results_queue, device):

    has_categorical_features = True if len(categorical_features) > 0 else False

    # Prepare data
    train_val_index, test_index = train_test_indices
    train_val_sites = site_stratify.index.values[train_val_index]
    test_sites = site_stratify.index.values[test_index]
    print(f"Fold {rank} - Train/Val sites: {train_val_sites} - Test sites: {test_sites}")
    site_stratify_train_val = site_stratify.loc[train_val_sites]
    train_sites, val_sites = train_test_split(
        site_stratify_train_val.index,
        stratify=site_stratify_train_val['stratify'],
        test_size=0.2,
        random_state=args.seed
    )
    data_train = data.loc[train_sites]
    data_val = data.loc[val_sites]
    data_test = data.loc[test_sites]

    writer = None

    # Create per-site stratification targets for inner CV
    site_stratify_cv = site_stratify.loc[train_sites]
    sites_cv = site_stratify_cv.index.values
    stratify_values_cv = site_stratify_cv['stratify'].values

    kf_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    cv_split = list(kf_cv.split(sites_cv, stratify_values_cv))

    train_ds_list = []
    val_ds_list = []
    for train_index_cv, val_index_cv in cv_split:
        train_sites_cv = sites_cv[train_index_cv]
        val_sites_cv = sites_cv[val_index_cv]

        data_train_cv = data.loc[train_sites_cv]
        data_val_cv = data.loc[val_sites_cv]

        train_stats_cv = compute_train_stats(data_train_cv, numerical_features)
        train_ds_cv = gpp_dataset(data_train_cv, train_stats_cv, numerical_features,
                                  categorical_features, test=False, permute=args.permute)
        val_ds_cv = gpp_dataset(data_val_cv, train_stats_cv, numerical_features,
                                categorical_features, test=True)

        train_ds_list.append(train_ds_cv)
        val_ds_list.append(val_ds_cv)

    trial_results = []

    # Hyperparameter tuning on inner CV loop
    for j in tqdm(range(args.num_trials), desc=f"Device {rank}"):
        try:
            hparams = sample_hyperparameters(hparam_space, args.model_type)
            total_cv_rmse = 0

            for k in range(3):
                train_ds_cv = train_ds_list[k]
                val_ds_cv = val_ds_list[k]

                g = torch.Generator()
                g.manual_seed(args.seed + rank + k)
                train_dl_cv = DataLoader(train_ds_cv, batch_size=hparams['batch_size'],
                                         shuffle=True, num_workers=0, generator=g,
                                         pin_memory=(device.type == 'cuda'))
                val_dl_cv = DataLoader(val_ds_cv, batch_size=1, shuffle=False,
                                       num_workers=0, pin_memory=(device.type == 'cuda'))

                model = make_model(args.model_type, len(numerical_features),
                                   conditional_dim, hparams, args)
                model = model.to(device=device)

                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=hparams['lr'],
                                             weight_decay=hparams['weight_decay'])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min',
                    patience=hparams['scheduler_patience'],
                    factor=hparams['scheduler_factor']
                )

                cv_fold_val_metrics, model = train_model(
                    train_dl_cv, val_dl_cv, model, optimizer, scheduler,
                    args.n_epochs, device, args.patience, writer,
                    args.early_stopping, has_categorical_features, args.seq2one
                )
                val_rmse_cv = cv_fold_val_metrics['rmse']
                total_cv_rmse += val_rmse_cv

            mean_val_rmse = total_cv_rmse / 3
            trial_results.append({"hparams": hparams, "mean_val_rmse": mean_val_rmse})  ### NEW
            print(f"Trial {j+1}/{args.num_trials} - fold {rank} - Hyperparameters: {hparams} - Mean Val RMSE: {mean_val_rmse:.4f}")

        except Exception as e:
            print(f"Error during trial {j+1} on fold {rank}: {e}")
            continue

    trial_results = sorted(trial_results, key=lambda r: r["mean_val_rmse"])
    top_k = trial_results[:3]  # keep best 3 configs
    print(f"Top {len(top_k)} hyperparameter sets for fold {rank}:")
    for r in top_k:
        print(f"  {r['hparams']} -> {r['mean_val_rmse']:.4f}")

    # Prepare final train/val sets
    writer = SummaryWriter(log_dir=os.path.join(base_path, 'runs', filename, f'fold_{rank}'))
    train_stats = compute_train_stats(data_train, numerical_features)
    train_ds = gpp_dataset(data_train, train_stats, numerical_features, categorical_features, test=False, permute=args.permute)
    val_ds = gpp_dataset(data_val, train_stats, numerical_features, categorical_features, test=True)

    final_metrics = None
    final_hparams = None
    best_model = None

    for candidate in trial_results:
        hparams = candidate["hparams"]
        print(f"[Fold {rank}] Trying final retrain with hparams: {hparams}")

        g = torch.Generator()
        g.manual_seed(args.seed + rank)
        train_dl = DataLoader(train_ds,
                            batch_size=hparams['batch_size'],
                            shuffle=True, num_workers=0, generator=g,
                            pin_memory=False)
        val_dl   = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False, num_workers=0, pin_memory=False)

        try:
            best_model = make_model(args.model_type, len(numerical_features),
                                    conditional_dim, hparams, args).to(device=device)

            optimizer = torch.optim.Adam(best_model.parameters(),
                                        lr=hparams['lr'],
                                        weight_decay=hparams['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min',
                patience=hparams['scheduler_patience'],
                factor=hparams['scheduler_factor']
            )

            best_val_metrics, best_model = train_model(
                train_dl, val_dl, best_model, optimizer, scheduler,
                args.n_epochs, device, args.patience, writer,
                args.early_stopping, has_categorical_features, args.seq2one
            )

            if np.isnan(list(best_val_metrics.values())).any():
                raise ValueError("NaN in validation metrics")

            final_metrics = best_val_metrics
            final_hparams = hparams
            print(f"[Fold {rank}] Final retrain succeeded with hparams {hparams}")
            break

        except Exception as e:
            print(f"[Fold {rank}] Final retrain failed for {hparams}: {e}")
            continue

    if final_metrics is None:
        print(f"[Fold {rank}] All top-{len(top_k)} configs failed. Skipping fold.")
        writer.close()
        results_queue.put((None, None, None, None))
        return

    # Save final model
    torch.save(best_model.state_dict(), os.path.join(base_path, "weights", f"{filename}_fold_{rank}.pt"))
    writer.close()

    # Final test set
    test_ds = gpp_dataset(data_test, train_stats, numerical_features, categorical_features, test=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    _, y_pred = test_loop(test_dl, best_model, device, has_categorical_features, args.seq2one)
    test_metrics, data_test_eval = evaluate_model(test_dl, y_pred)

    fold_df = data_test_eval[['TIMESTAMP', 'GPP_NT_VUT_REF', 'gpp_pred']]

    results_queue.put((fold_df, final_metrics, test_metrics, final_hparams))

    print(f"{args.model_type} test metrics - fold {rank} - R2: {test_metrics['r2']:.4f} | RMSE: {test_metrics['rmse']:.4f} | NMAE: {test_metrics['nmae']:.4f} | Abs Bias: {test_metrics['abs_bias']:.4f} | NSE: {test_metrics['nse']:.4f}")

def process_fold(rank, data, base_path, filename, args, numerical_features, categorical_features, conditional_dim, hparam_space, results_queue, folds, site_stratify):
    # Process a single fold in a separate process
    gpu_id = rank
    if args.device == 'cuda':
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)

    set_seed(args.seed)

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
    preds_filename = os.path.join(base_path, 'preds', f'{filename}.csv')
    df_out.to_csv(preds_filename)
    print(f'Predictions saved to {preds_filename}')
    print(f"{args.model_type} - Mean val R2: {np.mean(all_val_metrics['r2']):.4f} | Mean RMSE: {np.mean(all_val_metrics['rmse']):.4f} | Mean NMAE: {np.mean(all_val_metrics['nmae']):.4f} | Mean Abs Bias: {np.mean(all_val_metrics['abs_bias']):.4f} | Mean NSE: {np.mean(all_val_metrics['nse']):.4f}")
    print(f"{args.model_type} - Mean test R2: {np.mean(all_test_metrics['r2']):.4f} | Mean RMSE: {np.mean(all_test_metrics['rmse']):.4f} | Mean NMAE: {np.mean(all_test_metrics['nmae']):.4f} | Mean Abs Bias: {np.mean(all_test_metrics['abs_bias']):.4f} | Mean NSE: {np.mean(all_test_metrics['nse']):.4f}")
    metrics_total = compute_metrics(df_out['GPP_NT_VUT_REF'], df_out['gpp_pred'])
    print(f"{args.model_type} - Total R2: {metrics_total['r2']:.4f} | Total RMSE: {metrics_total['rmse']:.4f} | Total NMAE: {metrics_total['nmae']:.4f} | Total Abs Bias: {metrics_total['abs_bias']:.4f} | Total NSE: {metrics_total['nse']:.4f}")

def main(args):
    data_path = '../data/fdk_v342_ml.csv'
    data = pd.read_csv(data_path, index_col='sitename', parse_dates=['TIMESTAMP'])
    print("Using dataset:", data_path)
    numerical_features = ['TA_F_MDS', 'TA_DAY_F_MDS', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'VPD_DAY_F_MDS', 'PA_F', 'P_F', 'WS_F', 'FPAR']
    if args.model_type == 'LSTM' and args.extra_features:
        categorical_features = []
        numerical_features += ['whc']
    elif args.model_type == 'MLP' and args.extra_features:
        numerical_features += ['wscal']
        categorical_features = []
    else:
        categorical_features = []
    if args.modis:
        numerical_features += ['RED', 'NIR', 'BLUE', 'GREEN', 'SWIR1', 'SWIR2', 'SWIR3', 'LST_TERRA_Day_VZA0', 'LST_TERRA_Night_VZA0']
    print("Numerical features:", numerical_features)
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
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(kf.split(sites, stratify_values))
    
    base_path = "../"
    filename = f"{args.model_type}_global_model_{'extra_' if args.extra_features else ''}{'modis_' if args.modis else ''}{'attn_' if args.attention else ''}{'ln_' if args.layernorm else ''}{'permute_' if args.permute else ''}{'seq2one_' if args.seq2one else ''}{args.n_epochs}epochs_{args.num_trials}trials_seed{args.seed}_{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}"
    print(f"Saving to file: {filename}")
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
    parser.add_argument('-model', '--model_type', default='LSTM', type=str, choices=['LSTM', 'MLP', 'TCN'], help='Model type to use')
    parser.add_argument('-a', '--attention', action=argparse.BooleanOptionalAction, help='Whether to use attention mechanism in LSTM model')
    parser.add_argument('-ln', '--layernorm', action=argparse.BooleanOptionalAction, help='Whether to use layer normalization in LSTM model')
    parser.add_argument('-e', '--n_epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('-es', '--early_stopping', action=argparse.BooleanOptionalAction, default=True, help='Whether to use early stopping')
    parser.add_argument('-p', '--patience', default=10, type=int, help='Number of iterations (patience threshold) used for early stopping')
    parser.add_argument('-t', '--num_trials', default=20, type=int, help='Number of trials for hyperparameter tuning')
    parser.add_argument('-c', '--extra_features', action=argparse.BooleanOptionalAction, help='Whether to include extra features')
    parser.add_argument('-s', '--seed', default=31, type=int, help='Seed for reproducibility')
    parser.add_argument('-per', '--permute', action=argparse.BooleanOptionalAction, default=False, help='Whether to permute the data chunks in the dataset')
    parser.add_argument('-mod', '--modis', action=argparse.BooleanOptionalAction, default=False, help='Whether to include MODIS data')
    parser.add_argument('-s2o', '--seq2one', action=argparse.BooleanOptionalAction, default=False, help='Whether to use sequence-to-one prediction (only for LSTM)')
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
    print(f"> MODIS features: {args.modis}")
    print(f"> Permute data: {args.permute}")
    print(f"> Sequence-to-one prediction: {args.seq2one}")

    print(f"> Number of available devices: {torch.cuda.device_count()}")

    set_seed(args.seed)
    print(f"Seed set to {args.seed}")

    main(args)