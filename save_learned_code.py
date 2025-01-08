import numpy as np
from models.CM1DLayers import CorrelationMatrixLayer

init = 'checkpoints/good_checkpoints/coded_model-tmp.ckpt'

coding_mat = CorrelationMatrixLayer(init=init, get_from_model=True)

cmat = np.transpose(np.squeeze(coding_mat.cmat1D.weight.data.detach().clone().numpy()))

np.save(f'C:\\Users\\Patron\\PycharmProjects\\Indirect-Direct-ToF\\learned_codes\\{init.split('/')[-1]}', cmat)
