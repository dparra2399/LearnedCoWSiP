import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from dataset.dataset_utils import SimulatedLabelModule
from models.model_LIT_CODING import LITIlluminationModel
import pytorch_lightning as pl
import torch
import yaml


storage = "sqlite:///optuna_studies/illum_studies/study_illum_001.db"
config_file = 'config/average_configs/test_params_nt200.yaml'

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
    loss_id = config['loss_id']
    
    dataset_params = config['dataset']
    n_tbins = config['n_tbins']
    k = config['k']
    sigma = dataset_params['sigma']
    num_samples = dataset_params['num_samples']

    minmax_counts = dataset_params['minmax_counts']
    minmax_sbrs = dataset_params['minmax_sbrs']
    grid_size = dataset_params['grid_size']

    counts = torch.linspace(minmax_counts[0], minmax_counts[1], grid_size)
    sbrs = torch.linspace(minmax_sbrs[0], minmax_sbrs[1], grid_size)

class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def objective(trial):
    init_lr = trial.suggest_float("init_lr", 1e-5, 1e-1, log=True)
    lr_decay_gamma = trial.suggest_float("lr_decay_gamma", 1e-1, 1, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 128)
    epochs = trial.suggest_int("epochs", 10, 50)
    tv_reg = trial.suggest_float("tv_reg", 1e-5, 1e-1, log=True)
    beta = trial.suggest_int("beta", 1, 100)
    #num_samples = trial.suggest_int("num_samples", 4000, 32000)
    

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
            val_check_interval=0.5,
            max_epochs=epochs,
            logger=False,
            enable_checkpointing=False,
            callbacks=[OptunaPruning(trial, monitor="val_loss")],
        )
    else:
        trainer = pl.Trainer(
            max_epochs=epochs,
            val_check_interval=0.5,
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

    study.optimize(objective, n_trials=200)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    with open(f'config/average_configs/best_params_n{n_tbins}_k{k}.yaml', 'w+') as f:
        yaml.dump(study.best_params, f)