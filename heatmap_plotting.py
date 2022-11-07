import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg

currimg="5663ff96-2d67a04a_1"

attsc_layer2=[np.load("./snap/lastframe/heatmaps/onlyactions_loadpretrained2_noclweight_noreg_train_all_12h_9_5_5_bs8/" \
    + currimg + "/attention_scores/lang/layer_2/"+currimg+"_attsc_layer_2_lang_head_"+i+".npy") for i in range(nbheads)]

head_to_plot=1
source=attsc_layer2[head_to_plot-1]

plt.imshow(source, cmap='jet')
plt.colorbar()
plt.show()