import numpy as np
f = np.load("features/resnet_aug_features.npy")
l = np.load("features/resnet_aug_labels.npy", allow_pickle=True)
print(f.shape, len(l))

import numpy as np, os
print(os.path.exists("features/resnet_aug_features.npy"))
print(os.path.exists("features/resnet_aug_labels.npy"))