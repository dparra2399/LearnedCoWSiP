import numpy as np
from models.CM1DLayers import IlluminationLayer
import os
import yaml

init = 'n1024_k8_sigma10_photonstarved'
init = 'version_6'
path = os.path.join('experiments\\illum_models', init)

ckpt_path = os.path.join(path, 'checkpoints', 'coded_model.ckpt')
filename = init.split('/')[-1].split('.')[0]
folder = 'C:\\Users\\clwalker4\\PycharmProjects\\Indirect-Direct-ToF\\learned_codes'
with open(os.path.join(path, 'hparams.yaml'), 'r') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)
    n_tbins = hparams['n_tbins']
    k = hparams['k']
    loss_id = hparams['loss_id']

coding_mat = IlluminationLayer(n_tbins=n_tbins, k=k, init=ckpt_path, get_from_model=True)

cmat = np.transpose(np.squeeze(coding_mat.cmat1D.weight.data.detach().clone().numpy()))
illum = np.transpose(coding_mat.illumination.data.detach().numpy())

foldername = f'n{n_tbins}_k{k}_{loss_id}'
foldername = init
print(cmat.shape)
print(illum.shape)

os.makedirs(os.path.join(folder, 'bandlimited_models', foldername))

np.save(os.path.join(folder, 'bandlimited_models', foldername, 'coded_model'), cmat)
np.save(os.path.join(folder, 'bandlimited_models', foldername, 'illum_model'), illum)

