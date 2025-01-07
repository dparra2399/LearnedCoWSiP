import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from utils.model_utils import SimulatedDataModule
from pytorch_lightning.loggers import CSVLogger
from model_LIT_CODING import LITCodingModel
import pytorch_lightning as pl
import torch
import yaml

if_plot = True
rep_freq = 5 * 1e6
rep_tau = 1. / rep_freq
sigma = 10
n_tbins = 1024
k = 4

storage = "sqlite:///optuna_studies/study_1.db"
start_file = 'best_hyperparameters_3.yaml'

class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def objective(trial):
    init_lr = trial.suggest_float("init_lr", 1e-5, 1e-1, log=True)
    lr_decay_gamma = trial.suggest_float("lr_decay_gamma", 1e-1, 1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    epochs = trial.suggest_int("epochs", 50, 200)
    tv_reg = trial.suggest_float("tv_reg", 1e-5, 1e-1, log=True)
    beta = trial.suggest_int("beta", 1, 100)
    num_samples = trial.suggest_int("num_samples", 4000, 10000)
    num_counts = trial.suggest_int('counts' , 1, 30)
    
    counts = torch.linspace(10 ** 3, 10 ** 3, 10)
    sbr = torch.linspace(0.1, 0.1, 10)


    data_module = SimulatedDataModule(n_tbins, counts, sbr, rep_tau, batch_size, num_samples=num_samples, sigma=sigma,
                                      normalize=True)
    data_module.setup()

    lit_model = LITCodingModel(k=k, n_tbins=n_tbins, init_lr=init_lr, lr_decay_gamma=lr_decay_gamma,
                               beta=beta, tv_reg=tv_reg)

    # PyTorch Lightning Trainer with Optuna Pruning
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            devices=[0],
            accelerator='gpu',
            val_check_interval=0.25,
            max_epochs=epochs,
            logger=False,
            enable_checkpointing=False,
            callbacks=[OptunaPruning(trial, monitor="val_loss")],
        )
    else:
        trainer = pl.Trainer(
            max_epochs=epochs,
            val_check_interval=0.25,
            logger=False,
            enable_checkpointing=False,
            callbacks=[OptunaPruning(trial, monitor="val_loss")],
        )

    # Train the model
    trainer.fit(lit_model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

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

    with open(start_file, "r") as file:
        config = yaml.safe_load(file)

    study.enqueue_trial(config)  # Pre-tuned values

    study.optimize(objective, n_trials=200)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    with open('best_hyperparameters_4.yaml', 'w+') as f:
        yaml.dump(study.best_params, f)

    # {'init_lr': 0.0009145173790926249, 'lr_decay_gamma': 0.37425082035224766, 'batch_size': 24, 'epochs': 129, 'tv_reg': 0.09723338742824626, 'beta': 8, 'num_samples': 4661}