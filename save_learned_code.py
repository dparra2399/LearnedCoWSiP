import numpy as np
from models.CM1DLayers import IlluminationLayer
import os

init = 'checkpoints/coded_model-v5.ckpt'
filename = init.split('/')[-1].split('.')[0]
cmat_folder = 'C:\\Users\\Patron\\PycharmProjects\\Indirect-Direct-ToF\\learned_codes\\coding_matrices'
illum_folder = 'C:\\Users\\Patron\\PycharmProjects\\Indirect-Direct-ToF\\learned_codes\\illumination'


coding_mat = IlluminationLayer(init=init, get_from_model=True)

cmat = np.transpose(np.squeeze(coding_mat.cmat1D.weight.data.detach().clone().numpy()))
illum = np.squeeze(coding_mat.illumination.data.detach().numpy())

print(cmat.shape)
print(illum.shape)
np.save(os.path.join(cmat_folder, filename), cmat)
np.save(os.path.join(illum_folder, filename), illum)

