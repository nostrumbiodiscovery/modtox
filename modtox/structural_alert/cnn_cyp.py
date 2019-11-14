import istarmap
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, AveragePooling3D
from keras.layers import Dropout, Input, BatchNormalization
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
from rdkit import Chem
from rdkit.Chem import Descriptors
import modtox.constants.constants as cs
from itertools import count

def mae_to_sd(mae, schr=cs.SCHR, folder='.', output=None):
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

def extract_cnn_input(sdf):

    info = [];
    print("Load trajectory from {}".format(os.path.basename(sdf)))
    supl = Chem.SDMolSupplier(sdf)
    mols = [mol for mol in supl]
    getatoms = [mol.GetAtoms() for mol in mols]
    atoms = [[getatoms[i][x].GetSymbol() for x in range(len(getatoms[i]))] for i in range(len(getatoms))]

    print("Find CM of the ligand")

    conf = [mol.GetConformer() for mol in mols] # for each molecule in the sdf
    num_atoms = [co.GetNumAtoms() for co in conf] 
    pos = np.array([[list(co.GetAtomPosition(num)) for num in range(num_atoms[i])] for i,co in enumerate(conf)])
    masses = np.array([Descriptors.MolWt(mol) for mol in mols]) #masses of all the ligands
    centers_of_mass = [np.sum([np.array(getatoms[i][j].GetMass()) * pos[i][j] for j in range(num_atoms[i])], axis=0)/masses[i] for i in range(len(masses))]
    
    cm = np.mean(centers_of_mass, axis=0) # finally the mean cm

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

folder = "/home/moruiz/cyp/new_test_2/from_train/docking/"
dataset = "/home/moruiz/cyp/new_test_2/from_train/dataset/"
maes_lig = glob.glob(os.path.join(folder, "*dock_lib.maegz"))
maes_rec = glob.glob(os.path.join(folder, "*rec.maegz"))

actives = os.path.join(dataset, "actives.sdf")
inactives = os.path.join("/home/moruiz/cyp/dude/cp2c9/decoys_final.sdf")

#coverting ligands
if len(glob.glob(os.path.join(folder, "*dock_lib.sdf"))) == 0: sdfs_lig = [mae_to_sd(mae, folder=folder, output=None) for mae in maes_lig]
else: sdfs_lig = glob.glob(os.path.join(folder, "*dock_lib.sdf"))
                 
#coverting receptor
if len(glob.glob(os.path.join(folder, "*_rec.sdf"))) == 0: sdfs_rec = [mae_to_sd(mae, folder=folder, output=None) for mae in maes_rec]
else: sdfs_rec = glob.glob(os.path.join(folder, "*rec.sdf"))
sdfs = [[lig, rec] for lig, rec in zip(sdfs_lig, sdfs_rec)]

#Create vocabulary
vocabulary_elements, features = retrieve_vocabulary_from_sdf(sdfs_lig)
print(vocabulary_elements, features)

sdfs_lig = [ [sdf] for sdf in sdfs_lig]
cpus = 5
with Pool(cpus) as pool:
    result = list(tqdm(pool.istarmap(extract_cnn_input, sdfs_lig), total=len(sdfs_lig))) #[sdf, vectors]


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

import pdb; pdb.set_trace()
num, x, y, z, l = list(np.array(X).shape)
print(num, x, y, z, l)
X_train = X
y_train = keras.utils.to_categorical(Y, 2)
print("Build model")
model = Sequential()
input_layer = Input((x, y, z, l))

## convolutional layers
conv_layer1 = Conv3D(filters=18, kernel_size=(3, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=18, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)
average_layer = GlobalMaxPooling3D()(conv_layer2)

dense_layer2 = Dropout(0.4)(average_layer)
output_layer = Dense(units=2,  activation='softmax')(dense_layer2)

## define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(
  optimizer=keras.optimizers.Adam(lr=1e-2),
  loss=keras.losses.BinaryCrossentropy())

ratio = len([y for y in Y if y ==1]) / len(Y)

class_weight = {0: 1,
                1: ratio}

print("Fit model")
import pdb; pdb.set_trace()
model.fit(x=np.array(X_train), y=np.array(y_train), batch_size=128, epochs=50, validation_split=0.2, workers=4, class_weight=class_weight)
model.evaluate(X_train, y_train)
preds = model.predict(X_train)
import pdb; pdb.set_trace()



