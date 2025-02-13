import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from dataset.dataset_utils import SimulatedLabelModule
from models.model_LIT_CODING import LITIlluminationModel
import pytorch_lightning as pl
import torch
import yaml


storage = "sqlite:///optuna_studies/illum_studies/study_illum_peak_counts_002.db"
config_file = 'config/peak_configs/test_params_nt200_k4.yaml'

try:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    init_lr = config['init_lr']
    lr_decay_gamma = config['lr_decay_gamma']
    tv_reg = config['tv_reg']
    epochs = config['epochs']
    batch_size = config['batch_size']
    beta = config['beta']
    loss_id = config['loss_id']

    dataset_params = config['dataset']
    n_tbins = config['n_tbins']
    k = config['k']
    sigma = dataset_params['sigma']
    num_samples = dataset_params['num_samples']

    minmax_counts = dataset_params['minmax_counts']
    min_count = minmax_counts[0]
    max_counts = minmax_counts[1]
    minmax_sbrs = dataset_params['minmax_sbrs']
    grid_size = dataset_params['grid_size']

    print(minmax_counts)
    print(minmax_sbrs)
except (FileNotFoundError, TypeError) as e:
    print(e)
    exit(0)

class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def objective(trial):
    #init_lr = trial.suggest_float("init_lr", 1e-5, 1e-1, log=True)
    #lr_decay_gamma = trial.suggest_float("lr_decay_gamma", 1e-1, 1, log=True)
    #batch_size = trial.suggest_int("batch_size", 8, 128)
    #epochs = trial.suggest_int("epochs", 10, 50)
    #tv_reg = trial.suggest_float("tv_reg", 1e-5, 1e-1, log=True)
    #beta = trial.suggest_int("beta", 1, 100)
    #num_samples = trial.suggest_int("num_samples", 4000, 32000)
    #loss_id = trial.suggest_categorical('loss_id', ['mae', 'rsme', 'chardonnier'])
    #grid_size = trial.suggest_int("grid_size", 10, 30)
    min_count = trial.suggest_int("min_count", 10, 300)
    max_count = trial.suggest_int("max_count", min_count, 2000)

    counts = torch.linspace(min_count, max_count, grid_size)
    sbrs = torch.linspace(minmax_sbrs[0], minmax_sbrs[1], grid_size)

    label_module = SimulatedLabelModule(n_tbins, sources=counts, sbrs=sbrs, batch_size=batch_size,
                                        num_samples=num_samples)
    label_module.setup()

    lit_model = LITIlluminationModel(k=k, n_tbins=n_tbins, loss_id=loss_id, init_lr=init_lr, lr_decay_gamma=lr_decay_gamma,
                               beta=beta, tv_reg=tv_reg, sigma=sigma)

    # PyTorch Lightning Trainer with Optuna Pruning
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            devices=[0, 1],
            accelerator='gpu',
            val_check_interval=1.0,
            max_epochs=epochs,
            logger=False,
            enable_checkpointing=False,
            callbacks=[OptunaPruning(trial, monitor="val_loss")],
        )
    else:
        trainer = pl.Trainer(
            max_epochs=epochs,
            val_check_interval=1.0,
            logger=False,
            enable_checkpointing=False,
            callbacks=[OptunaPruning(trial, monitor="val_loss")],
        )

    # Train the model
    trainer.fit(lit_model, datamodule=label_module)

    # Return the last logged training loss
    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_float32_matmul_precision('medium')

    else:
        device = torch.device("cpu")


    study = optuna.create_study(study_name='my_study', storage=storage, load_if_exists=True,
                                direction="minimize")


    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    study.enqueue_trial(config)  # Pre-tuned values

    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    with open(f'config/average_configs/best_params_nt{n_tbins}_k{k}.yaml', 'a+') as f:
        yaml.dump(study.best_params, f)