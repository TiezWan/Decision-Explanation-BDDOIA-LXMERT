import numpy as np
import os


files = os.listdir('/root/dataset/NuScenes/feature_output_train/')
for i, file in enumerate(files):
    if file[-8:-4] is not 'info':
    print(file)
    break 
# #pass
featuress = []
    file_feature = np.load('/root/dataset/NuScenes/feature_output_train/'+file, allow_pickle=True)
    file_feature = file_feature.item()
    features = file_feature['feature']
    for i, feature in enumerate(features):
        feature = feature.cpu()
        featuress.append(feature)
    file_feature['feature'] = featuress
    np.save('/root/dataset/NuScenes/feature_output_train_backup/'+file, file_feature)

# a = F.item()
# b = a['feature']
# for i, b1 in enumerate(b):
#     b1 = b1.cpu()
# pass