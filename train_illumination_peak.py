from models.model_LIT_CODING import LITIlluminationPeakModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from dataset.dataset_utils import SimulatedLabelModule
import torch
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data directories")
    parser.add_argument("--yaml_file", type=str, help="Config file path", default='config/peak_configs/tester.yaml')
    parser.add_argument("--log_dir", type=str, help="Output model directory", default='experiments') 
    args = parser.parse_args()

    try:
        with open(args.yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        init_lr = config['init_lr']
        lr_decay_gamma = config['lr_decay_gamma']
        tv_reg = config['tv_reg']
        peak_reg = config['peak_reg']
        epochs = config['epochs']
        batch_size = config['batch_size']
        beta = config['beta']
        beta_max = config['beta_max']
        loss_id = config['loss_id']

        dataset_params = config['dataset']
        n_tbins = config['n_tbins']
        k = config['k']
        sigma = dataset_params['sigma']
        irf_filename = dataset_params['irf_filename']
        num_samples = dataset_params['num_samples']
        account_irf = dataset_params['account_irf']
        account_illum = dataset_params['account_illum']

        start_bin = dataset_params['start_bin']
        end_bin = dataset_params['end_bin']

        minmax_counts = dataset_params['minmax_counts']
        minmax_sbrs = dataset_params['minmax_sbrs']
        grid_size = dataset_params['grid_size']
        peak_factor = dataset_params['peak_factor']

        recon = config['recon']

        model_params = config['model_params']

        learn_illum = model_params['learn_illum']
        learn_coding_mat = model_params['learn_coding_mat']

        init_illum = model_params['init_illum']
        init_coding_mat = model_params['init_coding_mat']


        counts = torch.linspace(minmax_counts[0], minmax_counts[1], grid_size)
        sbrs = torch.linspace(minmax_sbrs[0], minmax_sbrs[1], grid_size)
    except (FileNotFoundError, TypeError) as e:
        print(e)
        exit(0)


    logger = CSVLogger(args.log_dir, name="illum_peak_models")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.log_dir}/{logger.name}/version_{logger.version}/checkpoints",
        filename ='coded_model',
        save_top_k=1,  # Save only the best model
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Minimize the monitored metric
    )

    label_module = SimulatedLabelModule(n_tbins, sources=counts, sbrs=sbrs, batch_size=batch_size,
                                        num_samples=num_samples, start_bin=start_bin, end_bin=end_bin)
    label_module.setup()

    print(len(label_module.train_dataset))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision('medium')

    else:
        device = torch.device("cpu")

    pl.seed_everything(42)

    trainer = pl.Trainer(logger=logger, max_epochs=epochs,
                          log_every_n_steps=250, val_check_interval=0.5,
                          callbacks=[checkpoint_callback])

    lit_model = LITIlluminationPeakModel(k=k, n_tbins=n_tbins, loss_id=loss_id, init_lr=init_lr, lr_decay_gamma=lr_decay_gamma,
                               beta=beta, beta_max=beta_max, peak_factor=peak_factor, peak_reg=peak_reg, tv_reg=tv_reg, sigma=sigma, recon=recon,
                               irf_filename=irf_filename, account_irf=account_irf, account_illum=account_illum,
                               init_coding_mat=init_coding_mat,learn_coding_mat=learn_coding_mat,
                               init_illum=init_illum,learn_illum=learn_illum)
    
    lit_model.save_hyperparameters({'dataset': dataset_params, 'epochs': epochs})
    torch.autograd.set_detect_anomaly(True)

    trainer.fit(lit_model, datamodule=label_module)