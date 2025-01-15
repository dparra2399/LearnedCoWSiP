from models.model_LIT_CODING import LITIlluminationModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from dataset.dataset_utils import SimulatedLabelModule
import torch
import pytorch_lightning as pl

import yaml

photon_count = 10 ** 3
sbr = 1.0

n_tbins = 1024
k = 4
sigma = 30

yaml_file = 'config/best_hyperparameters_tmp.yaml'


if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # Directory to save the model
        filename="coded_model",  # Base name for the checkpoint files
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
        print(e)
        exit(0)

    label_module = SimulatedLabelModule(n_tbins, batch_size=batch_size, num_samples=num_samples)
    label_module.setup()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_float32_matmul_precision('medium')

    else:
        device = torch.device("cpu")

    pl.seed_everything(42)

    logger = CSVLogger("tb_logs", name="my_model")

    trainer = pl.Trainer(logger=logger, max_epochs=epochs,
                          log_every_n_steps=250, val_check_interval=0.5,
                          callbacks=[checkpoint_callback])

    lit_model = LITIlluminationModel(k=k, n_tbins=n_tbins, init_lr=init_lr, lr_decay_gamma=lr_decay_gamma,
                               beta=beta, tv_reg=tv_reg, photon_count=photon_count, sbr=sbr, sigma=sigma)

    torch.autograd.set_detect_anomaly(True)

    trainer.fit(lit_model, datamodule=label_module)