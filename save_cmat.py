import numpy as np
from models.CM1DLayers import CorrelationMatrixLayer

init = 'checkpoints/coded_model-v8.ckpt'

coding_mat = CorrelationMatrixLayer(k=4, n_tbins=1024, init=init, get_from_model=True)

cmat = coding_mat.cmat1D.weight
cmat = np.transpose(cmat.detach().numpy().squeeze())

np.save(init.split('/')[-1], cmat)





