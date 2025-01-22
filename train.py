from models.model_LIT_CODING import LITCodingModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dataset.dataset_utils import SimulatedDataModule
import torch
import pytorch_lightning as pl

import yaml

rep_freq = 5 * 1e6
rep_tau = 1. / rep_freq
n_tbins = 1024
k=4
sigma = 20

counts = torch.linspace(10 ** 2, 10 ** 4, 10)
sbr = torch.linspace(0.1, 5.0, 10)


yaml_file = 'config/best_hyperparameters_3.yaml'
log_dir = 'experiments'


if __name__ == '__main__':
    logger = TensorBoardLogger(log_dir, name="code_models")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{log_dir}/{logger.name}/version_{logger.version}/checkpoints",  
        filename ='coded_model',
        save_top_k=1,  # Save only the best model
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Minimize the monitored metric
    )


    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        init_lr = config['init_lr']
        lr_decay_gamma = config['lr_decay_gamma']
        tv_reg = config['tv_reg']
        epochs = config['epochs']
        batch_size = config['batch_size']
        beta = config['beta']
        num_samples = config['num_samples']
    except (FileNotFoundError, TypeError) as e:
        pass

    data_module = SimulatedDataModule(n_tbins, counts, sbr, batch_size, num_samples=num_samples, sigma=sigma, normalize=True)
    data_module.setup()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_float32_matmul_precision('medium')

    else:
        device = torch.device("cpu")

    pl.seed_everything(42)

    trainer = pl.Trainer(logger=logger, max_epochs=epochs,
                          log_every_n_steps=250, val_check_interval=0.25,
                          callbacks=[checkpoint_callback])

    lit_model = LITCodingModel(k=k, n_tbins=n_tbins, init_lr=init_lr, lr_decay_gamma=lr_decay_gamma,
                               beta=beta, tv_reg=tv_reg)
    torch.autograd.set_detect_anomaly(True)

    trainer.fit(lit_model, datamodule=data_module)