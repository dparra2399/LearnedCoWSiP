import pytorch_lightning as pl
from dataset import SampleDataset
from torch.utils.data import DataLoader


class SimulatedDataModule(pl.LightningDataModule):
    def __init__(self,nt, nr, nc, photon_counts, sbr, train_samples=100, val_samples=10, batch_size=32):
        super().__init__()
        self.n_tbins = nt
        self.nr = nr
        self.nc = nc

        self.photon_counts = photon_counts
        self.sbr = sbr
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Called on every GPU
        # Set up train and validation datasets here
        self.train_dataset = SampleDataset(self.n_tbins, self.nr, self.nc, self.train_samples, self.photon_counts, self.sbr)
        self.val_dataset = SampleDataset(self.n_tbins, self.nr, self.nc, self.val_samples, self.photon_counts, self.sbr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
