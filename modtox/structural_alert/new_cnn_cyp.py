import istarmap
import argparse
import time
import collections
from rdkit import Chem
import keras
import json
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, AveragePooling3D
from keras.layers import Dropout, Input, BatchNormalization
from keras.models import model_from_yaml
from keras.models import load_model as keras_load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from keras.utils import multi_gpu_model
import h5py
import os
from multiprocessing import Pool
from itertools import repeat
import prody as pd
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import subprocess
import glob
from itertools import count
import tensorflow as tf
from tensorflow.python.client import device_lib
assert 'GPU' in str(device_lib.list_local_devices())
from keras import backend
backend.clear_session()
assert len(backend.tensorflow_backend._get_available_gpus()) > 0

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolfiles import MolToPDBFile
import sparse
import pandas


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_rec', action='store_true', help='Include ligands and receptor')
    parser.add_argument('--just_rec', action='store_true', help='Just receptor')
    parser.add_argument('--just_ligands', action='store_true', help='Just ligands')
    parser.add_argument('--volume', action='store_true', help='Include volume to atoms')
    args = parser.parse_args()
    print(args)
    return args.include_rec, args.just_rec, args.just_ligands, args.volume

def mae_to_sd(mae, schr=".", folder='.', output=None):
    if not output:
        output = os.path.splitext(os.path.basename(mae))[0]+".sdf"
    output = os.path.join(folder, output)
    sdconvert = os.path.join(schr, "utilities/sdconvert")
    command = "{} -imae {}  -osd {}".format(sdconvert, mae, output)
    print(command)
    subprocess.call(command.split())
    return output

def radius_assignation(folder='.', filename='radius.txt', costum=True):
    
    data = pandas.read_csv(os.path.join(folder, filename))[7:]
    radius = {}
    for i in data['Atomic']:
        if costum:
            if len(i.split()) == 7:
                radius[i.split()[1]] = float(i.split()[5])
            else:
                radius[i.split()[1]] = 1            
        else:
            radius[i.split()[1]] = 1
    return radius

def retrieve_vocabulary_from_sdf(sdfs):
    vocabulary_elements=[]; features=[]
    for sdf in tqdm(sdfs):
        supl = Chem.SDMolSupplier(sdf, removeHs=False)
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
        rear_back_pixel_center = self.rear_back_pixel_center = self.vertexes[0]  + np.array((self.resolution/2, self.resolution/2, self.resolution/2))
        self.n_pixels = n_pixels =  int(self.side / self.resolution)

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


def extract_cnn_input(sdf_lig, sdf_rec, center_of_mass, vocabulary_elements, features, resolution, volume=True,
                     include_rec=False, just_receptor=True, just_ligands=False):

    cm = center_of_mass # finally the mean cm
    side = 20
    print("CM", cm)
    print("Build grid")
    grid2 = Grid(cm, side, resolution)
    grid2.calculate_pixels()
    
    tot_atoms = []
    tot_mols = []
    
    if include_rec:
        sdfs = [sdf_lig, sdf_rec]
    if just_receptor:
        sdfs = [sdf_rec]
    if just_ligands:
        sdfs = [sdf_lig]

    for sdf in sdfs:
        print("Load trajectory from {}".format(os.path.basename(sdf)))
        supl = Chem.SDMolSupplier(sdf, removeHs=False)
        mols = [mol for mol in supl]
        getatoms = [mol.GetAtoms() for mol in mols]
        atoms = [[getatoms[i][x].GetSymbol() for x in range(len(getatoms[i]))] for i in range(len(getatoms))]
        conf = [mol.GetConformer() for mol in mols] # for each molecule in the sdf
        num_atoms = [co.GetNumAtoms() for co in conf]
        pos = np.array([[list(co.GetAtomPosition(num)) for num in range(num_atoms[i])] for i,co in enumerate(conf)])
        # now we only take atoms of ligand in a 20A radius (supposed to be all of them if small)
        us_atoms = [[atoms[j][i] for i in range(len(atoms[j])) if np.linalg.norm(pos[j][i]- cm) <= side ] for j in range(len(atoms))]
        tot_atoms += us_atoms
        tot_mols += mols
    print("Filling grid with atom info")
    atoms_all_mols, dict_at_pos, core_atoms, individuals = fill_grid_with_atoms(grid2, tot_mols, tot_atoms, cm, vocabulary_elements, features, side=side, 
                                                       volume=volume, include_rec=include_rec, just_receptor=just_receptor, 
                                                       just_ligands=just_ligands)
    return [atoms_all_mols, dict_at_pos, core_atoms, individuals, atoms, sdf_lig, sdf_rec]


def fill_grid_with_atoms(grid2, mols, atoms, cm, vocabulary_elements, features, side, include_rec=False, just_receptor=True, just_ligands=False, volume=True):

    at_all_mols = []
    dict_at_pos = {}
    core_atoms = {}
    individuals = {}
    print(len(mols), len(atoms))
    for mol in tqdm(range(len(mols))):
            sparse_matrix_dict = {}
            atoms_per_pixel = np.zeros((grid2.n_pixels, grid2.n_pixels, grid2.n_pixels, features),  dtype=int)
            molec = mols[mol]
            conf = molec.GetConformer() # for each molecule in the sdf
            num_atoms = conf.GetNumAtoms()
            coords_mol = np.array(list([conf.GetAtomPosition(num) for num in range(num_atoms)]))
            dict_at_pos[mol] = {}
            core_atoms[mol] = {}
            individuals[mol] = {}
            for atom in np.unique(atoms[mol]):
                dict_at_pos[mol][atom] = []
            for tt, atom in enumerate(range(len(atoms[mol]))):
                element = atoms[mol][atom]
                coords = coords_mol[atom]
                #assert grid2.is_point_inside(coords), coords
                ixs = np.array([int(round((x/grid2.resolution))) for x in np.array(np.array(coords) - np.array(grid2.rear_back_pixel_center))])
                #computing volumes
                if volume:
                    idxs = volume_occupancy(grid2.resolution, element, ixs, vocabulary_elements, side)
                else:
                    idxs = [ixs] 
                individuals[mol][tt] = idxs
                if len(dict_at_pos[mol][element]) == 0:
                    dict_at_pos[mol][element] = idxs
                    core_atoms[mol][element] = [ixs]
                else:
                    try:
                        dict_at_pos[mol][element] = np.concatenate((dict_at_pos[mol][element], idxs))
                        core_atoms[mol][element] =  np.concatenate((core_atoms[mol][element], [ixs]))
                    except ValueError:
                        print('ValueError', idxs)
                for idx in idxs:
                    try:
                        atoms_per_pixel[idx[0]][idx[1]][idx[2]]
                    except IndexError:
                        print('Cant assign atoms per pixel', idx)

                    pixel = atoms_per_pixel[idx[0]][idx[1]][idx[2]].copy()
                    try:
                        value =  pixel[vocabulary_elements[element]] + 1
                        pixel[vocabulary_elements[element]] = value
                    except TypeError:
                        value =  1
                        pixel[vocabulary_elements[element]] = value

                    atoms_per_pixel[idx[0], idx[1], idx[2]] = pixel
                    
                    index = (idx[0], idx[1], idx[2], vocabulary_elements[element])
                    sparse_matrix_dict[index]=value

            sparse_matrix_dict[(grid2.n_pixels-1,grid2.n_pixels-1,grid2.n_pixels-1,features-1)]=0
            try:
                x = sparse.COO(sparse_matrix_dict)
                at_all_mols.append(x)

            except ValueError:
                print('Probably negative indices detected')
                break
    print('moleculas', len(mols), len(core_atoms))
    return at_all_mols, dict_at_pos, core_atoms, individuals


def volume_occupancy(resolution, element, pixel, vocabulary_elements, side): 
    minindx = 0 #minimum index
    maxindx = side/resolution #maximum index

    radii = radius_assignation()
    radius = radii[element]
    #sphere equation (x-a)**2 + (y-a)**2 + (z-a)**2 = r**2
    maxrange = int(max((radius-resolution)/resolution, 0))
    rangex = list(range(pixel[0]-maxrange, pixel[0] + maxrange+1))
    rangey = list(range(pixel[1]-maxrange, pixel[1] + maxrange+1))
    rangez = list(range(pixel[2]-maxrange, pixel[2] + maxrange+1))
    #computing activated pixels
    usefuls = []
    
    for i in range(len(rangex)):
        for j in range(len(rangey)):
            for k in range(len(rangez)):
                if (i*resolution)**2 + (j*resolution)**2 + (k*resolution)**2 <= radius**2:
                    usefuls += [[a,b,c] for a in [-i,i] for b in [-j,j] for c in [-k,k]]
                else:
                    break
    Vol_pixel = resolution**3
    Vol_element = 4/3*np.pi*radius**3
    usefuls = np.unique(usefuls, axis=0)
    movedp = [pixel + idx for idx in usefuls]
    moved = [ids for ids in movedp if not ids[ids<minindx].any() if not ids[ids>=maxindx].any()]
    if len(moved) == 0:
        print('Ep! atom outside box!')
    Aprox_vol = Vol_pixel*len(moved)
    Error_vol = (abs(Aprox_vol-Vol_element)/Vol_element)*100
  #  print('Real vol', Vol_element)
  #  print('Calculate vol', Aprox_vol)
  #  print('Error vol (%)', Error_vol)
    return moved


def pixels_importance(model, X, X_rec, y_train, include_rec, nsample):
    
    print('Analysis on pixels...')
    print('SAMPLE', i)
    x,y,z,l = X[0].shape
    nsamples = len(X)
    blank_pixel = np.zeros(l)

    params_test = {'dim': (x,y,z),
              'batch_size': 1,
              'n_classes': 2,
              'n_channels': l,
              'shuffle': False,
              'tofit': True}

    print('Preparing generator')
    test_generator = DataGenerator([nsample], labels=y_train, X=X, **params_test)
    pred_baseline = model.predict_generator(test_generator)[0]
    print('pred baseline', pred_baseline)
    true_label = y_train[nsample]
    print('true label', true_label)
    label = np.argmax(pred_baseline)

    pos = []
    old_X = X.copy()
    old_X_rec = X_rec.copy()
    ligand_contributions = []
    for coordx in tqdm(range(x)):
        for coordy in range(y):
            for coordz in range(z):
                new_X_train = old_X[nsample].todense()
                if 1 in new_X_train[coordx,coordy,coordz]:
                    new_X_train[coordx, coordy, coordz] = blank_pixel
                    new_X_sparse = [sparse.COO(new_X_train)]
                    test_generator = DataGenerator([0], labels=y_train, X=new_X_sparse, **params_test)
                    pred =  model.predict_generator(test_generator)[0]
                    if true_label[label] == 1:
                        importance_single_pixel = pred[1] - pred_baseline[1]
                    else:
                        importance_single_pixel = pred_baseline[1] - pred[1]
                    pos.append([coordx, coordy, coordz, importance_single_pixel])

                    #if we have the receptor included:
                    if include_rec:
                        X_rec = old_X_rec[nsample].todense()
                        if X_rec[coordx,coordy,coordz] != new_X_train[coordx,coordy,coordz]:
                            ligand_contributions.append([coordx,coordy,coordz])
                else:
                    importance_single_pixel=0

    print('ligand', len(ligand_contributions))
    print('rec + ligand', len(pos))

    return np.array(pos), ligand_contributions


def pixels_per_atom(model, X, y_train, indiv_positions, cores, nsample, idx_receptor=None):

    if idx_receptor is None:
        atoms_pos = indiv_positions[nsample] 
    else:
        atoms_pos = indiv_positions[idx_receptor] 
    atoms_pos = checking_positions(atoms_pos)
    print('Analysis on pixels...')
    print('Len x', len(X))
    x,y,z,l = X[0].shape
    print('Dimensions', x,y,z,l)
    params_test = {'dim': (x,y,z),
              'batch_size': 1,
              'n_classes': 2,
              'n_channels': l,
              'shuffle': False,
              'tofit': True}

    print('Preparing generator')
    test_generator = DataGenerator([nsample], labels=y_train, X=X, verbose=0, **params_test)
    pred_baseline = model.predict_generator(test_generator)[0]
    print('pred baseline', pred_baseline)
    true_label = y_train[nsample]
    print('true label', true_label)
    label = np.argmax(pred_baseline)

    nsamples = len(X)
    x,y,z,l = X[0].shape
    blank_pixel = np.zeros(l)

    pos = []
    old_X = X.copy()
    for i, at in enumerate(tqdm(atoms_pos.keys())): #for each atom
        positions = atoms_pos[at] #positions of that atom
        new_X_train = old_X[nsample].todense()
        for posi in positions: # deactivating each position
            coordx = posi[0]; coordy = posi[1]; coordz = posi[2]
            if 1 in new_X_train[coordx,coordy,coordz]:
                new_X_train[coordx,coordy,coordz] = blank_pixel
        new_X_sparse = [sparse.COO(new_X_train)]
        test_generator = DataGenerator([0], labels=y_train, X=new_X_sparse, verbose=0, **params_test)
        pred =  model.predict_generator(test_generator)[0]
        if true_label[label] == 1:
            importance_single_atom = pred[1] - pred_baseline[1]
        else:
            importance_single_atom = pred_baseline[1] - pred[1]
        centx = cores[i][0]; centy = cores[i][1]; centz = cores[i][2];
        pos.append([centx, centy, centz, importance_single_atom])
                
    return np.array(pos), atoms_pos



def checking_positions(posit):
    
    '''saving positions in a dict if inside the box'''
    
    inside_box_positions = []
    for i in posit.keys():
        if len(posit[i]) > 0:
            inside_box_positions.append(i)
    return {i:j for i,j in zip(posit.keys(), posit.values()) if i in inside_box_positions}

def customizing_pdbs(clusters=list(range(10)), nsample=[0]):
    
    total_idx = 0
    for k, (sdf_lig, sdf_rec) in enumerate(zip(sdfs_lig, sdfs_rec)): #for each sdf file
        print('\t-----> cluster', k, 'how many in cluster')
        supl_lig = Chem.SDMolSupplier(sdf_lig, removeHs=False)
        supl_rec = Chem.SDMolSupplier(sdf_rec, removeHs=False)
        mols_lig = [mol for mol in supl_lig]
        idx_receptor = len(mols_lig) + total_idx #index of the receptor in indiv_positions
        mols_rec = [mol for mol in supl_rec]
        if k in clusters:
            for i in range(len(dict_core_pos[k])): #for each molecule of that file
                print('\t----->> molecule', i, 'of: ', len(dict_core_pos[k]))
                if not i == len(dict_core_pos[k])-1 and i in nsample:
                    central_atoms = np.concatenate([i for i in dict_core_pos[k][i].values()])
                    central_atoms_rec = np.concatenate([i for i in dict_core_pos[k][len(dict_core_pos[k])-1].values()])
                    ff_lig = MolToPDBFile(mols_lig[i], 'clust_{}_{}_lig.pdb'.format(k,i))
                    ff_rec = MolToPDBFile(mols_rec[0], 'clust_{}_{}_rec.pdb'.format(k,i))
                    real_idx = i + total_idx
                    poses_lig, atoms_pos_lig = pixels_per_atom(loaded_model, X, y_train, indiv_positions, central_atoms, real_idx)

                    poses_rec, atoms_pos_rec = pixels_per_atom(loaded_model, X, y_train, indiv_positions, central_atoms_rec, real_idx, idx_receptor)
                    ranged_lig = heatmap_on_betafactors(file='clust_{}_{}_lig.pdb'.format(k,i), vector=poses_lig[:,3], filled=atoms_pos_lig)
                    ranged_rec = heatmap_on_betafactors(file='clust_{}_{}_rec.pdb'.format(k,i), vector=poses_rec[:,3], ranged=None, filled=atoms_pos_rec)  
        #update index referenced to total molecules (mols of ligand + mol of receptor)
        total_idx += len(mols_lig) + 1

def rescale_vector(x, ranged=None):

    tmax = 100
    tmin = -100
    rmax = max(x)
    rmin = min(x)
    if ranged is not None:
        if ranged[1] > rmax: rmax = ranged[1]
        if ranged[0] < rmin: rmin = ranged[0]
    return np.array([((m - rmin)/(rmax - rmin))*(tmax-tmin) + tmin for m in x]), (rmin, rmax)


def heatmap_on_betafactors(file, vector, ranged=None, filled=None):
    toreplace = []
    values = []
    with open (file, 'r') as f:
        filedata = f.readlines()
        lines = [i.split() for i in filedata]
        k = 0
        for j, line in enumerate(lines):
            if line[0]=='HETATM':
                toreplace.append(j)
                #filling the mnolecules out of the box with 0's
                if j not in filled.keys():
                    values.append(0.00)
                else:
                    values.append(vector[k])
                    k+=1 
    rescaled, (rmin, rmax) = rescale_vector(values, ranged)
    toreplace = np.array(toreplace)
    newdata = ''
    j = 0
    for i,dr in enumerate(filedata):
        if i in toreplace:
            try:
                newdata += dr.replace(' 0.00', ' ' + str(round(rescaled[j], 2)))
            except IndexError:
                assert False
            j+=1
        else:
            newdata += dr         
    newfile = file.split('.')[0] + '_mod.pdb'
    with open(newfile, 'w') as ff:
        ff.write(newdata)
        print('pdb written on', newfile)
    return (rmin, rmax)



####################################

restart = False

folder = "/scratch/jobs/moruiz/structural_alert/"
actives = os.path.join(folder, "actives.sdf")
inactives = os.path.join(folder, "decoys_final.sdf")
resolution = 0.5
include_rec, just_rec, just_ligands, volume = arg_parser()
folder_tosave = 'topass_05_2_rec_imp'
if not os.path.exists(folder_tosave): os.mkdir(folder_tosave)

if not restart:
    sdfs_lig = glob.glob(os.path.join(folder, "input__cluster.c*__dock_lib.sdf")) 
    sdfs_rec = glob.glob(os.path.join(folder, "input__cluster.c*__rec.sdf"))
    sdfs_lig = np.sort(sdfs_lig)
    sdfs_rec = np.sort(sdfs_rec)
    if include_rec:
        sdfs = np.concatenate((sdfs_lig, sdfs_rec))
    if just_rec:
        sdfs = sdfs_rec
    if just_ligands:
        sdfs = sdfs_lig
    #Create vocabulary
    vocabulary_elements, features = retrieve_vocabulary_from_sdf(sdfs)

    print(vocabulary_elements, features)
    np.save(os.path.join(folder_tosave, 'vocabulary'), vocabulary_elements)    
    np.save(os.path.join(folder_tosave, 'features'), features)
    np.save(os.path.join(folder_tosave, 'sdfs_lig'), sdfs_lig)    
    np.save(os.path.join(folder_tosave, 'sdfs_rec'), sdfs_rec)    

if restart:
    vocabulary_elements = np.load(os.path.join(folder_tosave,'vocabulary.npy'), allow_pickle=True).item()
    features = np.load(os.path.join(folder_tosave,'features.npy')).item()
    sdfs_lig = np.load(os.path.join(folder_tosave,'sdfs_lig.npy'))
    sdfs_rec = np.load(os.path.join(folder_tosave,'sdfs_rec.npy'))
    if include_rec:
        sdfs = np.concatenate((sdfs_lig, sdfs_rec))
    if just_rec:
        sdfs = sdfs_rec
    if just_ligands:
        sdfs = sdfs_lig
    print(vocabulary_elements, features, sdfs_lig)
#Create COM
coms = []
for sdf in tqdm(sdfs):
    print(sdf)
    supl = Chem.SDMolSupplier(sdf, removeHs=False)
    mols = [mol for mol in supl]
    getatoms = [mol.GetAtoms() for mol in mols]
    atoms = [[getatoms[i][x].GetSymbol() for x in range(len(getatoms[i]))] for i in range(len(getatoms))]
    conf = [mol.GetConformer() for mol in mols] # for each molecule in the sdf
    num_atoms = [co.GetNumAtoms() for co in conf]
    pos = np.array([[list(co.GetAtomPosition(num)) for num in range(num_atoms[i])] for i,co in enumerate(conf)])
    masses = np.array([Descriptors.MolWt(mol) for mol in mols]) #masses of all the ligands
    centers_of_mass = [np.sum([np.array(getatoms[i][j].GetMass()) * pos[i][j] for j in range(num_atoms[i])], axis=0)/masses[i] for i in range(len(masses))]
    coms.extend(centers_of_mass)
center_grid = np.mean(np.array(coms), axis=0)
print('center of grid', center_grid)
result = [extract_cnn_input(sdf_lig, sdf_rec, center_grid, vocabulary_elements, features, resolution, volume=volume, 
          include_rec=include_rec, just_receptor=just_rec, just_ligands=just_ligands) for sdf_lig, sdf_rec in zip(sdfs_lig, sdfs_rec)]


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


X = []; Y = []; sdfs = []; total_names=[]; total_indices=[]; dict_core_pos = []; indiv_positions={}; dict_atoms_pos = []
tot = 0
prev_max = -1

for clust, d , atoms, indiv , _, sdf_lig, sdf_rec in tqdm(result): # for each cluster
    ii = 0
    if just_ligands:
        sdfs.append(sdf_lig)
    if include_rec:
        sdfs.append(sdf_lig)
        sdfs.append(sdf_rec)
    dict_atoms_pos.append(d)
    dict_core_pos.append(atoms)
    for key in indiv.keys():
        new_key = key + prev_max+1
        indiv_positions[new_key] = indiv[key]
    prev_max = max(indiv_positions.keys())
    with open(sdf_lig, "r") as f:
        data = f.readlines()
        ids = [i+1 for i, j in zip(count(), data) if j == '$$$$\n']
        del ids[-1]
        ids.insert(0, 0) #Insert first line as always have a molecule name
        ids = np.array(ids)
        names_lig = [data[idx].split('\n')[0] for idx in ids]
        print('names', len(names_lig))
        if "clust3" in sdf_lig:
            assert names[0] == "CHEMBL115876"
            assert names[-1] == "CHEMBL255389"
            assert names[3] == "ZINC36928916"
    for i, x in tqdm(enumerate(clust)): # for each mol in the cluster
        total_indices.append(i)
        if include_rec: 
            end = len(clust) - 1 #idx of receptor
            x_receptor = clust[len(clust) - 1] 
        else: 
            end = len(clust) + 1 
        if i != end:
            if names_lig[i] in acts:
                Y.append(1)
                if include_rec:   
                    x = x_receptor + x
                X.append(x)
                ii += 1
            elif names_lig[i] in inacts:
                Y.append(0)
                if include_rec:   
                    x = x_receptor + x
                X.append(x)
                ii += 1
            elif names_lig[i] not in acts and names_lig[i] not in inacts:
                Y.append(1)
                if include_rec:   
                    x = x_receptor + x
                X.append(x)
                ii += 1;

    total_names.extend(names_lig)
    print(sdf_lig, ii, len(names_lig), len(total_indices), len(total_names))

    y_train = keras.utils.to_categorical(Y, 2)

n = len(X)
x,y,z,l = X[0].shape
print('train shape', y_train.shape, len(dict_core_pos), n,x,y,z,l)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, X, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, tofit=True):
        self.dim = dim
        self.batch_size = batch_size
        self.y = labels
        self.X = X
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tofit = tofit
        self.on_epoch_end()

    def __len__(self):
        if self.tofit:
            return int(np.floor(len(self.list_IDs) / self.batch_size)) 
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size)) + 1
    def __getitem__(self, index):
        
        # Generate indexes of the batch
        
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        try:
            X = [self.X[k].todense() for k in list_IDs_temp]
        except IndexError:
            print('list ids', list_IDs_temp)
            print('len(x)', len(self.X))
            print( self.X)
        y = [self.y[k, :] for k in list_IDs_temp]
        
        return np.array(X), np.array(y)
        
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

params_train = {'dim': (x,y,z),
          'batch_size': 10,
          'n_classes': 2,
          'n_channels': l,
          'shuffle': False,
          'tofit': True}

params_validation = {'dim': (x,y,z),
          'batch_size': 10,
          'n_classes': 2,
          'n_channels': l,
          'shuffle': False,
          'tofit': True}

params_test = {'dim': (x,y,z),
          'batch_size': 10,
          'n_classes': 2,
          'n_channels': l,
          'shuffle': False,
          'tofit': False}

train_idx = np.random.choice(list(range(len(y_train))), size=int(0.8*len(y_train)), replace=False)
val_idx = [idx for idx in list(range(len(y_train))) if idx not in train_idx]
test_idx = list(range(len(y_train)))


if not restart:
    tf.set_random_seed(0)
    print("Build model")
    
    print("Build model")
    model = Sequential()
    input_layer = Input((x,y,z,l)) 
    
    ## convolutional layers
    conv_layer1 = Conv3D(filters=32, kernel_size=(2, 2, 2), padding="same", activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=32, kernel_size=(2, 2, 2),padding="same", activation='relu')(conv_layer1)
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)
    
    ## add max pooling to obtain the most imformatic features
    conv_layer3 = Conv3D(filters=64, kernel_size=(2, 2, 2),padding="same", activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=64, kernel_size=(2, 2, 2),padding="same", activation='relu')(conv_layer3)
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)
    
    
    ## add max pooling to obtain the most imformatic features
    conv_layer5 = Conv3D(filters=128, kernel_size=(2, 2, 2), padding="same",activation='relu')(pooling_layer2)
    conv_layer6 = Conv3D(filters=128, kernel_size=(2, 2, 2),padding="same", activation='relu')(conv_layer5)
    pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer6)
    
    ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
    pooling_layer4 = BatchNormalization()(pooling_layer3)
    flatten_layer = Flatten()(pooling_layer4)
    
    ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
    ## add dropouts to avoid overfitting / perform regularization
    #dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(flatten_layer)
    dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=2, activation='softmax')(dense_layer2)
    
    
    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(loss='binary_crossentropy', optimizer=Adadelta(lr=0.1), metrics=['accuracy'])
    
    ## Define y1/y0 ratio
    weight_for_0 = (1 / len([y for y in Y if y == 0]))*(len(Y))/2.0 
    weight_for_1 = (1 / len([y for y in Y if y == 1]))*(len(Y))/2.0
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(class_weight)
    
    print('train val test, len(x)', len(train_idx), len(val_idx), len(test_idx), len(X))
    training_generator = DataGenerator(train_idx, labels=y_train, X=X, **params_train)
    validation_generator = DataGenerator(val_idx, labels=y_train, X=X, **params_validation)
    
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min', verbose=1)
  #  checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')
    
    
    print("Fit model")
    history = History()
    #model.fit_generator(generator=training_generator, validation_data=validation_generator,  class_weight=class_weight, epochs=20, callbacks = [early_stop,checkpoint, reduce_lr_loss])
    model.fit_generator(generator=training_generator, validation_data=validation_generator,  class_weight=class_weight, epochs=20, callbacks = [early_stop, reduce_lr_loss])
    #model.fit_generator(generator=training_generator, validation_data=validation_generator,  class_weight=class_weight, pickle_safe=True, epochs=40)
    print('Model fitted!') 
    test_generator = DataGenerator(test_idx, labels=y_train, X=X, **params_test)
    ac1 = model.evaluate_generator(generator=test_generator)
    preds1 = model.predict_generator(test_generator)
    Y = [np.argmax(y) for y in y_train]
    print(confusion_matrix(Y, [np.argmax(pred) for pred in preds1]))
    model.save(os.path.join(folder_tosave, 'all_model.h5'))
    
    print("Saved model to disk")    

else:
    tf.set_random_seed(0) 
    loaded_model = keras_load_model(os.path.join(folder_tosave, 'all_model.h5'))
    loaded_model.compile(loss='binary_crossentropy', optimizer=Adadelta(lr=0.1), metrics=['accuracy'])
    test_generator = DataGenerator(test_idx, labels=y_train, X=X, **params_test)
    ac1 = loaded_model.evaluate_generator(generator=test_generator)
    print(ac1)
    preds1 = loaded_model.predict_generator(test_generator)
    Y = [np.argmax(y) for y in y_train]
    print('conf matrix', confusion_matrix(Y, [np.argmax(pred) for pred in preds1]))

    model = loaded_model
#############################################

print('POSTPROCESS')
analysis_pixels = False
analysis_element = True
importance_element = False


if importance_element:
    print('Sensitivty analysis')
    
    importances_dict = vocabulary_elements.copy()
    elem_dict = {}
    for element in tqdm(vocabulary_elements.keys()):
        sites = []
        totmol = 0
        for clust in range(len(dict_atoms_pos)):
            nums = dict_atoms_pos[clust].keys()
            for num in nums:
                elems = dict_atoms_pos[clust][num].keys()
                if element in elems:
                    idx_el = (totmol, dict_atoms_pos[clust][num][element])
                    sites.append(idx_el)
                totmol += 1
        elem_dict[element] = sites
    
    average_elements = vocabulary_elements.copy()
    tot_molecs = np.sum([len(j.keys()) for j in dict_atoms_pos])
    for i in vocabulary_elements.keys():
        total_element = np.sum([len(elem_dict[i][j][1]) for j in range(len(elem_dict[i]))])
        print(i, total_element)
        average_elements[i] = total_element/tot_molecs
    
    print('Average elements', average_elements)
    
    for element in vocabulary_elements.keys():
        print(element)
        old_X = X.copy()
        channel = vocabulary_elements[element]
        new_X_train = np.zeros((len(X),x,y,z,l))
        for n in tqdm(range(len(X))):
            new_X_train[n] = X[n].todense()
        
        #all sites with this element 
        molec = [i for (i,j) in elem_dict[element]]
        pos = [j for (i,j) in elem_dict[element]]
        for at, coo in tqdm(zip(molec, pos), total = len(molec)):
            for idx in coo:
                i = idx[0];  j = idx[1];  k = idx[2]
                if not new_X_train[at, i, j, k, channel] != 0:
                    new_X_train[at, i, j, k, channel] = X[at].todense()[i,j,k,channel]
                new_X_train[at,i,j,k,channel] = 0
        
        print('Building new sparsed matrix')        
        new_X_sparse = sparse.COO(new_X_train)
    
        #Calculating importance            
        print('Testing...')
        test_generator = DataGenerator(test_idx, labels=y_train, X=new_X_sparse, **params_test)
        
        print('Computing performances')
        ac2 = model.evaluate_generator(generator=test_generator)
        print(ac1, ac2)
        importance = ac2[1]-ac1[1] 
        importances_dict[element] =  importance
    
        print('import dic', importances_dict)
        
        with open(os.path.join(folder_tosave, 'importances.json'), 'w') as fp:
            json.dump(importances_dict, fp)

if analysis_pixels:

    for i in range(len(X)):
        pos, ligand_contributions = pixels_importance(model, X, X_rec, y_train, include_rec, nsample=i) 
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        np.save(os.path.join(folder_tosave, 'pixels_imp_{}_{}'.format(i, resolution)), pos)
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=pos[:,3], cmap='viridis')

if analysis_element:
    side = 20
    cm = center_grid
    print("CM", cm)
    print("Build grid")
    grid2 = Grid(cm, side, resolution)
    grid2.calculate_pixels()
    if include_rec:
        sdfs_rec = [s for i,s in enumerate(sdfs) if i%2 != 0]
        sdfs_lig = [s for i,s in enumerate(sdfs) if i%2 == 0]
   
    customizing_pdbs(clusters=[0], nsample=[4])


assert 1==0

print('Analysis per element')

nsample=0
sample = X.copy()[nsample]
xs, ys, ls, _ = sample.shape
key = "N"; 
channel = vocabulary_elements[key]

idxs_dict = vocabulary_elements.copy();

#First prediction
pred = model.predict(np.array(sample).reshape(1, 20, 20, 20, 9))[0]
label_pred = np.argmax(pred)
prob = pred[label_pred]


# CHeck atoms with that channel
idxs = []
for r in range(xs):
    for k in range(ys):
        for m in range(ls):
            if sample[r, k, m, channel] != 0:
                print("A")
                idxs.append([r,k,m])
                idxs_dict[key] = idxs
                print(idxs_dict)


# Remove one atom at a time
importances_per_atom = []
for idxs in idxs_dict[key]:
    x,y,z = idxs
    position = rear_bck_pixel + np.array([+x+1,-y,-z])
    print(position)
    new_sample = X_train.copy()[nsample]
    new_sample[x,y,z,channel] = 0
    pred = model.predict(np.array(new_sample).reshape(1, 20, 20, 20, 9))[0]
    importance = prob - pred[label_pred]
    importances_per_atom.append(importance)
importances_per_atom


print('End normally!')
