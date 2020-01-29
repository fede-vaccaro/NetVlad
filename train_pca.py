import h5py
from sklearn.decomposition import PCA

from netvlad_model import NetVLADSiameseModel
import open_dataset_utils as my_utils
import paths
import numpy as np
from tqdm import tqdm
my_model = NetVLADSiameseModel()
vgg_netvlad = my_model.build_netvladmodel()
weight_name = "model_e111_sc-adam-2_0.0686.h5"

print("Loading weights: " + weight_name)
vgg_netvlad.load_weights(weight_name)

generator = my_utils.LandmarkTripletGenerator(paths.landmarks_path, model=my_model.get_netvlad_extractor(),
                                              use_multiprocessing=True)
custom_generator = generator.generator()

a = []
p = []
n = []

for i in tqdm(range(500)):
    x, _ = next(custom_generator)
    a_, p_, n_ = vgg_netvlad.predict(x)
    a += [a_]
    p += [p_]
    n += [n_]

generator.loader.stop_loading()

a = np.vstack((d for d in a)).astype('float32')
p = np.vstack((d for d in n)).astype('float32')
n = np.vstack((d for d in n)).astype('float32')

descs = np.vstack((a, p, n))
print(descs.shape)
del a, p, n

print("Computing PCA")
pca = PCA(512)
pca.fit(descs)
del descs

pca_dataset = h5py.File("pca.h5", 'w')
pca_dataset.create_dataset('components', data=pca.components_)
pca_dataset.create_dataset('mean', data=pca.mean_)
pca_dataset.create_dataset('explained_variance', data=pca.explained_variance_)
pca_dataset.close()
