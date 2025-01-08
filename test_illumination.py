import numpy as np

from models.CM1DLayers import IlluminationLayer, ZNCCLayer
from dataset.dataset import SampleLabels
from utils.torch_utils import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
#
if_plot = True

rep_freq = 5 * 1e6
rep_tau = 1. / rep_freq
sigma = 10
# counts = torch.linspace(10 ** 2, 10 ** 6, 20)
# sbr = torch.linspace(0.05, 10.0, 20)

photon_count = 10**3
sbr = 1.0
n_tbins = 1024
k = 4


#init = 'checkpoints/good_checkpoints/coded_model-v9.ckpt'
inits = ['TruncatedFourier', 'checkpoints/coded_model.ckpt']

sample_data = SampleLabels(n_tbins, num_samples=1024)
num_samples = len(sample_data)
labels = sample_data.labels

gt_depths = bin2depth(labels, num_bins=n_tbins, tau=rep_tau)



if if_plot:
    fig, axs = plt.subplots(len(inits), 3)


counter = 0
names = []
for init in inits:
    if init.endswith('.ckpt'): get_from_model = True
    else: get_from_model = False

    model = IlluminationLayer(k=k, n_tbins=n_tbins, init=init, get_from_model=get_from_model)

    zncc_layer = ZNCCLayer()

    c_vals = model(labels, photon_count, sbr)

    pred_depths = zncc_layer(c_vals, model.cmat1D.weight.data.detach().clone())

    pred_depths = bin2depth(pred_depths, num_bins=n_tbins, tau=rep_tau)
    #
    loss = torch.mean(torch.abs(pred_depths.squeeze() - gt_depths.squeeze()))
    #loss = criterion_RMSE(pred_depths, gt_depths)

    print(f'MAE {init.split('/')[-1].split('.')[0]}: {loss * 1000:.3f} mm ')
    names.append(init.split('/')[-1].split('.')[0])
    if if_plot:
        cmat = model.cmat1D.weight
        cmat = np.transpose(cmat.detach().numpy().squeeze())

        illum = model.illumination.numpy().squeeze()
        axs[counter][0].plot(illum)
        axs[counter][0].set_ylabel(init.split('/')[-1].split('.')[0])
        axs[counter][0].set_title('Illumination')
        axs[counter][1].plot(cmat)
        #axs[counter][1].set_title(init.split('/')[-1].split('.')[0])

        axs[0][2].bar(counter, loss, label=f'{loss * 1000:.2f}mm')
        axs[0][2].set_title('MAE (Lower Better)')
        axs[0][2].set_ylabel('MAE (mm)')

        if counter != 0:
            axs[counter][2].axis('off')
    counter += 1

axs[0][2].set_xticks(np.arange(0, len(inits)))
axs[0][2].set_xticklabels(names)
axs[0][2].legend()



plt.show(block=True)
print('hello world')

