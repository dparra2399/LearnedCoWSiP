import numpy as np
from models.CM1DLayers import IlluminationLayer
import os
import yaml

init = r'experiments/illum_models/version_5'
filename = init.split('/')[-1].split('.')[0]
cmat_folder = 'C:\\Users\\Patron\\PycharmProjects\\Indirect-Direct-ToF\\learned_codes\\coding_matrices'
illum_folder = 'C:\\Users\\Patron\\PycharmProjects\\Indirect-Direct-ToF\\learned_codes\\illumination'

with open(os.path.join(init, 'hparams.yaml'), 'r') as f:
    config = yaml.safe_load(f)
    n_tbins = config['n_tbins']

coding_mat = IlluminationLayer(n_tbins=n_tbins, init=os.path.join(init, r'checkpoints/coded_model.ckpt'), get_from_model=True)

cmat = np.transpose(np.squeeze(coding_mat.cmat1D.weight.data.detach().clone().numpy()))
illum = np.transpose(coding_mat.illumination.data.detach().numpy())

print(cmat.shape)
print(illum.shape)
np.save(os.path.join(cmat_folder, filename), cmat)
np.save(os.path.join(illum_folder, filename), illum)

