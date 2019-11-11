# %%
import os

import numpy as np
from keras.applications.resnet import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def create_image_dict(img_list):
    input_shape = (224, 224, 3)
    tensor = {}
    for path in img_list:
        img = image.load_img(path, target_size=(input_shape[0], input_shape[1]))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img_key = path.strip('holidays_small/')
        tensor[img_key] = img
    # tensor = np.array(tensor)
    return tensor


# %%

img_dict = create_image_dict(get_imlist('holidays_small'))
img_dict.keys()

# %%
# create dataset
queries = []
positives = []
negatives = []

triplets_file = open("triplets.dat", "r")
for line in triplets_file.readlines():
    split = line.split(" ")[:3]

    queries.append(img_dict[split[0]])
    positives.append(img_dict[split[1]])
    negatives.append(img_dict[split[2]])

queries = np.array(queries)
positives = np.array(positives)
negatives = np.array(negatives)

# %%
# create model
from keras.applications import ResNet50

embedding_size = 128

input_shape = (224, 224, 3)

resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)

for layer in resnet.layers:
    layer.trainable = False
    # print(layer, layer.trainable)

resnet.layers.pop(0)

# %%
# set layers untrainable
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

input_q = Input(shape=(224, 224, 3))
input_p = Input(shape=(224, 224, 3))
input_n = Input(shape=(224, 224, 3))

resnet_q = resnet(input_q)
resnet_p = resnet(input_p)
resnet_n = resnet(input_n)

resnet_output = resnet.output_shape[1]
embedding = Dense(embedding_size, input_shape=(resnet_output,), activation='relu', name="embedding1")

# %%
embedding_q = embedding(resnet_q)
embedding_p = embedding(resnet_p)
embedding_n = embedding(resnet_n)

resnet_qpn = Model([input_q, input_p, input_n], [embedding_q, embedding_p, embedding_n])

# %%

plot_model(resnet_qpn, to_file='base_network.png', show_shapes=True, show_layer_names=True)
# resnet_qpn.summary()

result = resnet_qpn.predict([queries[:1], positives[:1], negatives[:1]])

# %%

all_data_len = len(img_dict.keys())
n_train = 1491

fake_true_pred = np.zeros((n_train, embedding_size * 3))
fake_true_pred_val = np.zeros((all_data_len - n_train, embedding_size * 3))

queries_train = queries[:n_train]
positives_train = positives[:n_train]
negatives_train = negatives[:n_train]

queries_test = queries[n_train:]
positives_test = positives[n_train:]
negatives_test = negatives[n_train:]

# %%

from triplet_loss import TripletLossLayer

batch_size = 128
epochs = 12

# train session
opt = Adam(lr=0.0001)  # choose optimiser. RMS is good too!

loss_layer = TripletLossLayer(alpha=1., name='triplet_loss_layer')(resnet_qpn.output)
resnet_qpn = Model(inputs=resnet_qpn.input, outputs=loss_layer)
resnet_qpn.compile(optimizer=opt)

# %%

filepath = "trip_Holyday_v1_{epoch:02d}_BS%d.hdf5" % batch_size
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=25)
callbacks_list = [checkpoint]

#model_json = resnet_qpn.to_json()
#with open("resnet_qpn.json", "w") as json_file:
#    json_file.write(model_json)


H = resnet_qpn.fit(
    x=[queries_train, positives_train, negatives_train],
    y=None,
    batch_size=batch_size,
    epochs=epochs,
    #validation_data=([queries_test, positives_test, negatives_test], None),
    verbose=1,
    #callbacks=callbacks_list
)

resnet_qpn.save_weights("model.h5")
print("Saved model to disk")


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.plot(H.history['loss'], label='training loss')
plt.plot(H.history['val_loss'], label='validation loss')
plt.legend()
plt.title('Train/validation loss')
plt.show()