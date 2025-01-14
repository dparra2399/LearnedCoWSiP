import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from dataset.dataset_utils import SimulatedLabelModule
from models.model_LIT_CODING import LITIlluminationModel
import pytorch_lightning as pl
import torch
import yaml


sigma = 10
n_tbins = 1024
k = 4

photon_count = 10 ** 3
sbr = 1.0

storage = "sqlite:///optuna_studies/study_illumination_002.db"
start_file = 'config/best_hyperparameters_tmp.yaml'
#start_file = None

class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def objective(trial):
    init_lr = trial.suggest_float("init_lr", 1e-5, 1e-1, log=True)
    lr_decay_gamma = trial.suggest_float("lr_decay_gamma", 1e-1, 1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    epochs = trial.suggest_int("epochs", 50, 200)
    tv_reg = trial.suggest_float("tv_reg", 1e-5, 1e-1, log=True)
    tv_reg_illum = trial.suggest_float("tv_reg_illum", 1e-5, 1e-1, log=True)
    beta = trial.suggest_int("beta", 1, 100)
    num_samples = trial.suggest_int("num_samples", 40000, 400000)
    


    label_module = SimulatedLabelModule(n_tbins, batch_size=batch_size, num_samples=num_samples)
    label_module.setup()

    lit_model = LITIlluminationModel(k=k, n_tbins=n_tbins, init_lr=init_lr, lr_decay_gamma=lr_decay_gamma,
                               beta=beta, tv_reg=tv_reg, tv_reg_illum=tv_reg_illum, photon_count=photon_count, sbr=sbr)

    # PyTorch Lightning Trainer with Optuna Pruning
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            devices=[0],
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


    if start_file is not None:
        with open(start_file, "r") as file:
            config = yaml.safe_load(file)
        study.enqueue_trial(config)  # Pre-tuned values

    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    with open('config/best_hyperparameters_illumination_v1.yaml', 'w+') as f:
        yaml.dump(study.best_params, f)