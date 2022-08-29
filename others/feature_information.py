import numpy as np


feature = np.load('/root/dataset/NuScenes/feature_train/n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801727862460.npy', allow_pickle=True)

feature = feature.item()
featuress = featuress.item()
features = feature['feature']
featuresss = featuress['feature']

pass