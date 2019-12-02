import istarmap
#from rdkit import Chem
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, AveragePooling3D
from keras.layers import Dropout, Input, BatchNormalization
from keras.models import model_from_yaml
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model
from sklearn.model_selection import train_test_split
import h5py
import os
from multiprocessing import Pool
from itertools import repeat
import prody as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from tqdm import tqdm
import subprocess
import glob
from itertools import count
from tensorflow.python.client import device_lib
assert 'GPU' in str(device_lib.list_local_devices())
from keras import backend
assert len(backend.tensorflow_backend._get_available_gpus()) > 0


def mae_to_sd(mae, schr=".", folder='.', output=None):
    if not output:
        output = os.path.splitext(os.path.basename(mae))[0]+".sdf"
    output = os.path.join(folder, output)
    sdconvert = os.path.join(schr, "utilities/sdconvert")
    command = "{} -imae {}  -osd {}".format(sdconvert, mae, output)
    print(command)
    subprocess.call(command.split())
    return output

def retrieve_vocabulary_from_sdf(sdfs):
    vocabulary_elements=[]; features=[]
    for sdf in tqdm(sdfs):
        supl = Chem.SDMolSupplier(sdf)
        mols = [mol for mol in supl]
        getatoms = [mol.GetAtoms() for mol in mols]
        elements = set([getatoms[i][x].GetSymbol() for i in range(len(getatoms)) for x in range(len(getatoms[i]))])
        for element in elements:
            if element not in vocabulary_elements:
                vocabulary_elements.append(element)
    vocabulary_elements = {res:i for i, res in enumerate(vocabulary_elements)}
    features = len(vocabulary_elements)
    return  vocabulary_elements, features

class Grid(object):

    def __init__(self, center, side, resolution=None):
        self.center = center
        self.side = side
        self.resolution = resolution
        self.vertexes = self._calculate_vertexes()
    def _calculate_vertexes(self):
        center = self.center
        side = self.side

        vertexes = []
        v1 = center + [-side/2, -side/2, -side/2]
        v2 = center + [side/2, -side/2, -side/2]
        v3 = center + [-side/2, -side/2, side/2]
        v4 = center + [-side/2, side/2, -side/2]
        v5 = center + [side/2, side/2, -side/2]
        v6 = center + [side/2, -side/2, side/2]
        v7 = center + [-side/2, side/2, side/2]
        v8 = center + [side/2, side/2, side/2]
        vertexes.extend([v1, v2, v3, v4, v5, v6, v7, v8])
        return vertexes
    
    def calculate_pixels(self):
        print("Calculating pixels...")
        rear_back_pixel_center = self.rear_back_pixel_center = self.vertexes[6]  + np.array((self.resolution/2, -self.resolution/2, -self.resolution/2))
        self.n_pixels = n_pixels =  int(self.side / self.resolution)
        current_pixel_center = rear_back_pixel_center; self.pixels = np.empty((n_pixels, n_pixels, n_pixels),  dtype=object)
        for z in range(n_pixels):
            for j in range(n_pixels):
                for  i in range(n_pixels):
                    self.pixels[i][j][z] = Grid(current_pixel_center, self.resolution)
                    current_pixel_center = current_pixel_center + np.array((self.resolution, 0, 0))
                current_pixel_center = rear_back_pixel_center + np.array((0, (j+1)*-self.resolution, (z+1)*-self.resolution))



    def is_point_inside(self, point):
        dist = np.linalg.norm(self.center-point)
        dist_center_diagonal = np.linalg.norm(self.center-self.vertexes[0])
        if dist < dist_center_diagonal:
            return True
        else:
            return False

    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        for vertex in self.vertexes:
            vx, vy,vz = vertex.tolist()
            ax.scatter(vx, vy, vz)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return fig.show()

def extract_cnn_input(sdf, center_of_mass, vocabulary_elements, features):

    info = [];
    print("Load trajectory from {}".format(os.path.basename(sdf)))
    supl = Chem.SDMolSupplier(sdf)
    mols = [mol for mol in supl]
    getatoms = [mol.GetAtoms() for mol in mols]
    atoms = [[getatoms[i][x].GetSymbol() for x in range(len(getatoms[i]))] for i in range(len(getatoms))]
    conf = [mol.GetConformer() for mol in mols] # for each molecule in the sdf
    num_atoms = [co.GetNumAtoms() for co in conf]
    pos = np.array([[list(co.GetAtomPosition(num)) for num in range(num_atoms[i])] for i,co in enumerate(conf)])

    print("Find CM of the ligand")

    
    cm = center_of_mass # finally the mean cm

    side = 20
    print("CM", cm)

    print("Build grid")
    grid2 = Grid(cm, side, 1)
    grid2.calculate_pixels()

    print("Checking created grid")
    inside = [grid2.is_point_inside for pixel in grid2.pixels]
    assert  all(inside), "Some point of the grid out of the box check code!"

    # now we only take atoms of ligand in a 20A radius (supposed to be all of them if small)

    lim_dist = 20
    us_atoms = [[atoms[j][i] for i in range(len(atoms[j])) if np.linalg.norm(pos[j][i]- cm) <= lim_dist ] for j in range(len(atoms))]
    
    print("Filling grid with atom info")
    atoms_all_mols = fill_grid_with_atoms(grid2, mols, us_atoms, cm, vocabulary_elements, features)
    print(len(atoms_all_mols), sdf)
    return [atoms_all_mols, sdf]

def fill_grid_with_atoms(grid2, mols, atoms, cm, vocabulary_elements, features):

    at_all_mols = []
    for mol in range(len(mols)):
        atoms_per_pixel = np.zeros((grid2.n_pixels, grid2.n_pixels, grid2.n_pixels, features),  dtype=int)
        conf = mols[mol].GetConformer() # for each molecule in the sdf
        num_atoms = conf.GetNumAtoms()
        coords_mol = np.array(list([conf.GetAtomPosition(num) for num in range(num_atoms)]))

        for atom in range(len(atoms[mol])):
            element = atoms[mol][atom]
            coords = coords_mol[atom]

            assert grid2.is_point_inside(coords), coords

            idxs = np.array([int(abs((x/grid2.resolution))) for x in np.array(np.array(coords) - np.array(grid2.rear_back_pixel_center))])
            #If atom is outside grid
            try:
                atoms_per_pixel[idxs[0]][idxs[1]][idxs[2]]
            except IndexError:
                continue

            pixel = atoms_per_pixel[idxs[0]][idxs[1]][idxs[2]].copy()
            try:
                pixel[vocabulary_elements[element]] += 1
            except TypeError:
                pixel[vocabulary_elements[element]] = 1

            atoms_per_pixel[idxs[0], idxs[1], idxs[2]] = pixel
        at_all_mols.append(atoms_per_pixel)
    return at_all_mols

####################################

restart = True

if restart:
    X_train = np.load("Xtrain.npy", allow_pickle=True)
    y_train = np.load("Ytrain.npy", allow_pickle=True)
    Y = [np.argmax(y) for y in y_train]
    num, x, y, z, l = list(np.array(X_train).shape)
else:
    folder = "/home/moruiz/cyp/new_test_2/from_train/docking/"
    dataset = "/home/moruiz/cyp/new_test_2/from_train/dataset/"
    maes_lig = glob.glob(os.path.join(folder, "*dock_lib.maegz"))
    maes_rec = glob.glob(os.path.join(folder, "*rec.maegz"))
    
    actives = "actives.sdf"
    inactives = "inactives.sdf"
    
    #coverting ligands
    if len(glob.glob("*dock_lib.sdf")) == 0: sdfs_lig = [mae_to_sd(mae, folder=folder, output=None) for mae in maes_lig]
    else: sdfs_lig = glob.glob("*dock_lib.sdf")
                     
    #coverting receptor
    if len(glob.glob("*_rec.sdf")) == 0: sdfs_rec = [mae_to_sd(mae, folder=folder, output=None) for mae in maes_rec]
    else: sdfs_rec = glob.glob("*rec.sdf")
    sdfs = [[lig, rec] for lig, rec in zip(sdfs_lig, sdfs_rec)]
    
    #Create vocabulary
    vocabulary_elements, features = retrieve_vocabulary_from_sdf(sdfs_lig)
    print(vocabulary_elements, features)
    
    #Create COM
    coms = []
    for sdf in sdfs_lig:
        supl = Chem.SDMolSupplier(sdf)
        mols = [mol for mol in supl]
        getatoms = [mol.GetAtoms() for mol in mols]
        atoms = [[getatoms[i][x].GetSymbol() for x in range(len(getatoms[i]))] for i in range(len(getatoms))]
        conf = [mol.GetConformer() for mol in mols] # for each molecule in the sdf
        num_atoms = [co.GetNumAtoms() for co in conf]
        pos = np.array([[list(co.GetAtomPosition(num)) for num in range(num_atoms[i])] for i,co in enumerate(conf)])
        masses = np.array([Descriptors.MolWt(mol) for mol in mols]) #masses of all the ligands
        centers_of_mass = [np.sum([np.array(getatoms[i][j].GetMass()) * pos[i][j] for j in range(num_atoms[i])], axis=0)/masses[i] for i in range(len(masses))]
        coms.extend(centers_of_mass)
    print(coms)
    center_grid = np.mean(np.array(coms), axis=0)
    
    assert center_grid.shape[0] == 3
    
    
    
    sdfs_lig = [ sdf for sdf in sdfs_lig]
    cpus = 5
    iterable = zip(sdfs_lig, repeat(center_grid))
    #with Pool(cpus) as pool:
        #result = list(tqdm(pool.istarmap(extract_cnn_input, iterable), total=len(sdfs_lig))) #[sdf, vectors]
    result = [extract_cnn_input(sdf, center_grid, vocabulary_elements, features) for sdf in sdfs_lig]
    
    with open(actives, "r") as f:
        data = f.readlines()
        ids = np.array([i+1 for i, j in zip(count(), data) if j == '$$$$\n'])
        ids[-1] = 0 # first line contains a name, but not the last
        acts = [data[idx].split('\n')[0] for idx in ids] 
    with open(inactives, "r") as f:
        data = f.readlines()
        ids = np.array([i+1 for i, j in zip(count(), data) if j == '$$$$\n'])
        ids[-1] = 0
        inacts = [data[idx].split('\n')[0] for idx in ids] 
    
    X = []; Y = []
    tot = 0
    
    for clust, sdf in tqdm(result): # for each cluster
        ii = 0
        toremove = []
        with open(sdf, "r") as f:
            data = f.readlines()
            ids = np.array([i+1 for i, j in zip(count(), data) if j == '$$$$\n'])
            ids[-1] = 0 # first line contains a name, but not the last
            names = [data[idx].split('\n')[0] for idx in ids]
        for i,x in tqdm(enumerate(clust)): # for each mol in the cluster 
            if names[i] in acts:
                Y.append(1)
                X.append(x)
                ii += 1; 
            if names[i] in inacts:
                Y.append(0)
                X.append(x)
                ii += 1; 
            if names[i] not in acts and names[i] not in inacts: 
                print('Unexpected name (usually cluster...)', names[i])
                toremove.append(names[i])
        for rm in toremove: names.remove(rm)
        print(sdf, ii, len(names))
    
    num, x, y, z, l = list(np.array(X).shape)
    print(num, x, y, z, l)
    
    X_train = X
    y_train = keras.utils.to_categorical(Y, 2)
    
    print(y_train)

    np.save("Xtrain", np.array(X_train))
    np.save("Ytrain", np.array(y_train))

print(X_train.shape, y_train.shape)

from keras.utils import multi_gpu_model
print("Build model")
model = Sequential()
#model = multi_gpu_model(model, gpus=2)
input_layer = Input((x, y, z, l))

## convolutional layers
conv_layer1 = Conv3D(filters=32, kernel_size=(2, 2, 2), padding="same", activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=32, kernel_size=(2, 2, 2),padding="same", activation='relu')(conv_layer1)
pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

## add max pooling to obtain the most imformatic features
conv_layer3 = Conv3D(filters=64, kernel_size=(2, 2, 2),padding="same", activation='relu')(pooling_layer1)
conv_layer4 = Conv3D(filters=64, kernel_size=(2, 2, 2),padding="same", activation='relu')(conv_layer3)
pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)


## add max pooling to obtain the most imformatic features
conv_layer5 = Conv3D(filters=256, kernel_size=(2, 2, 2), padding="same",activation='relu')(pooling_layer2)
conv_layer6 = Conv3D(filters=256, kernel_size=(2, 2, 2),padding="same", activation='relu')(conv_layer5)
pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer6)

## perform batch normalization on the convolution outputs before feeding it to MLP architecture
pooling_layer4 = BatchNormalization()(pooling_layer3)
flatten_layer = Flatten()(pooling_layer4)

## create an MLP architecture with dense layers : 4096 -> 512 -> 10
## add dropouts to avoid overfitting / perform regularization
#dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
#dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=512, activation='relu')(flatten_layer)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=2, activation='softmax')(dense_layer2)


## define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])

## Define y1/y0 ratio
ratio = len([y for y in Y if y == 1]) / len([y for y in Y if y == 0])
class_weight = {0: 1,
                1: ratio}

assert ratio == 1.1669394435351883

history = History()
earlyStopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('model_save.hdf5', save_best_only=True, monitor='loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

print("Fit model")
history = model.fit(x=np.array(X_train), y=np.array(y_train), batch_size=128, callbacks=[history, earlyStopping, mcp_save, reduce_lr_loss], epochs=50, class_weight=class_weight)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
