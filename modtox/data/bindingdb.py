import os
import numpy as np
from rdkit import DataStructs
from tqdm import tqdm
from rdkit import Chem
import modtox.Helpers.preprocess as pr
import modtox.Helpers.formats as ft


class BindingDB():

    def __init__(self, bindingdb, train, test, folder_to_get='.', n_molecules_to_read=None, folder_output='.',
                 production=False, debug=False):
        self.production = production
        self.sdf = os.path.abspath(bindingdb)
        self.used_mols = 'used_mols.txt'
        self.test = test
        self.debug = debug
        self.production = production
        self.train = train
        self.folder_output = folder_output
        self.folder_to_get = folder_to_get
        self.n_molecules_to_read = n_molecules_to_read
        self.reading_from_bindingdb()

    def split_sdf(self, readfromfile=True):

        self.active_inchi, self.active_names = self.filtering(self.active_inchi, self.active_names)
        self.inactive_inchi, self.inactive_names = self.filtering(self.inactive_inchi, self.inactive_names)
        if not os.path.exists(self.folder_output):
            os.mkdir(self.folder_output)
        self.cleaning()
        w = Chem.SDWriter(os.path.join(self.folder_output, "actives.sdf"))
        for m in self.active_mols:
            w.write(m)

        w = Chem.SDWriter(os.path.join(self.folder_output, "inactives.sdf"))
        for m in self.inactive_mols:
            w.write(m)

        return os.path.join(self.folder_output, "actives.sdf"), os.path.join(self.folder_output, "inactives.sdf"), len(
            self.active_mols), len(self.active_mols)

    def filtering(self, iks, which_names):
        # checking duplicates
        uniq = set(iks)
        if len(iks) > len(uniq):  # cheking repeated inchikeys
            print('Duplicates detected')
            indices = {value: [i for i, v in enumerate(iks) if v == value] for value in uniq}
            iks = [iks[x[0]] for x in indices.values()]  # filtered inchikeys
            which_names = [which_names[x[0]] for x in indices.values()]  # filtering ids: we only get the first
        else:
            print('Duplicates not detected')

        return iks, which_names

    def similarity_calc(self, train, test):
        # use when excesive memory usage
        noaccep = []
        print('Computing similarities...')
        for idx, fp_test in enumerate(tqdm(test)):
            accep = True
            while accep:
                for i, fp_train in enumerate(train):
                    sim = DataStructs.FingerprintSimilarity(fp_train, fp_test)
                    if sim > 0.7:
                        noaccep.append(idx)
                        accep = False
                accep = False
        return noaccep

    def cleaning(self, cut=0.7):
        # recording instances from the training data
        if self.train:
            with open(os.path.join(self.folder_output, self.used_mols), 'w+') as r:
                for item in [*self.active_inchi, *self.inactive_inchi]:
                    r.write("{}\n".format(item))

        # extracting molecules from test already present in train
        if self.test:
            with open(os.path.join(self.folder_to_get, self.used_mols), 'r') as r:
                data = r.readlines()
                datalines = [x.split('\n')[0] for x in data]
                # filtering by exact inchikey
                use_active = [ind for ind, inchi in enumerate(self.active_inchi) if inchi not in datalines]
                use_inactive = [ind for ind, inchi in enumerate(self.inactive_inchi) if inchi not in datalines]
                active_inchi = np.array(self.active_inchi)[use_active]
                inactive_inchi = np.array(self.inactive_inchi)[use_inactive]
                self.active_mols = np.array(self.active_mols)[use_active]
                self.inactive_mols = np.array(self.inactive_mols)[use_inactive]
                # filtering by similarity >= 0.7
                mols_test_active = [Chem.inchi.MolFromInchi(str(inchi)) for inchi in active_inchi]
                mols_test_inactive = [Chem.inchi.MolFromInchi(str(inchi)) for inchi in inactive_inchi]
                mols_train = [Chem.inchi.MolFromInchi(str(inchi)) for inchi in datalines]
                fps_test_active = [Chem.Fingerprints.FingerprintMols.FingerprintMol(m) for m in mols_test_active]
                fps_test_inactive = [Chem.Fingerprints.FingerprintMols.FingerprintMol(m) for m in mols_test_inactive]
                fps_train = []
                print('Computing fingerprints train..')
                for mol in tqdm(mols_train):
                    try:
                        fps_train.append(Chem.Fingerprints.FingerprintMols.FingerprintMol(mol))
                    except:
                        pass
                print('Similarities analysis...')
                for k, fps_tar in enumerate([fps_test_active, fps_test_inactive]):
                    idxs = []
                    for i, fp1 in tqdm(enumerate(fps_tar), total=len(fps_tar)):
                        for fp2 in fps_train:
                            if DataStructs.FingerprintSimilarity(fp1, fp2) > cut:
                                idxs.append(i)
                                break
                    print('Detected ', len(idxs), ' similarities')
                    if k == 0:
                        good_idxs = [i for i in range(len(mols_test_active)) if i not in idxs]
                        self.active_inchi = np.array(active_inchi)[good_idxs]
                        self.active_mols = self.active_mols[good_idxs]
                    else:
                        good_idxs = [i for i in range(len(mols_test_inactive)) if i not in idxs]
                        self.inactive_inchi = np.array(inactive_inchi)[good_idxs]
                        self.inactive_mols = self.inactive_mols[good_idxs]

                print('Active before cleaning: {}, after: {}'.format(len(active_inchi), len(self.active_inchi)))
                print('Inactive before cleaning: {}, after: {}'.format(len(inactive_inchi), len(self.inactive_inchi)))
        return

    def reading_from_bindingdb(self):

        suppl = Chem.SDMolSupplier(self.sdf)
        mols = [mol for mol in suppl if mol]
        ic = [mol.GetProp('IC50 (nM)') for mol in mols]
        inchis = [mol.GetProp('Ligand InChI') for mol in mols]
        names = [mol.GetProp('_Name') for mol in mols]
        print('Read', len(mols))
        novalued = []
        ic_clean = []
        if self.debug: ic = ic[:3]
        for j, i in enumerate(ic):
            try:
                i = float(i)
                ic_clean.append(i)
            except ValueError:
                if len(i) == 0:
                    novalued.append(j)
                elif i[0] == '>':
                    if float(i.split('>')[1]) >= 10000:
                        i = 10001  # adding to inactives
                        ic_clean.append(float(i))
                        pass
                    else:
                        i = 5000  # adding to inconclusives
                        ic_clean.append(float(i))
                        pass
                elif i[0] == '<':
                    if float(i.split('<')[1]) <= 1000:
                        i = 999  # adding to inactives
                        ic_clean.append(float(i))
                        pass
                    else:
                        i = 5000  # adding to inconclusives
                        ic_clean.append(float(i))
                        pass
        del ic
        # filtering actives and inactives
        # setting thresholds
        # actives if IC < 1000 nM; inconclusives if 1000<IC<10000; inactives if IC > 10000
        actives = [i for i, j in enumerate(ic_clean) if float(j) <= 1000]
        inactives = [i for i, j in enumerate(ic_clean) if float(j) >= 10000]
        inconclusives = [i for i, j in enumerate(ic_clean) if 1000 < float(j) < 10000]

        self.active_inchi = np.array(inchis)[actives]
        self.inactive_inchi = np.array(inchis)[inactives]
        self.active_mols = np.array(mols)[actives]
        self.inactive_mols = np.array(mols)[inactives]

        self.active_names = []
        self.inactive_names = []
        i = 0
        for mol in self.active_mols:
            name = 'bin_{}'.format(i)
            mol.SetProp('_Name', name)
            self.active_names.append(name)
            #   if len(mol.GetProp('_Name')) < 1: mol.SetProp('_Name', str(i))
            #   self.active_names.append(mol.GetProp('_Name'))
            i += 1
        for mol in self.inactive_mols:
            name = 'bin_{}'.format(i)
            mol.SetProp('_Name', name)
            self.inactive_names.append(name)
            # if len(mol.GetProp('_Name')) < 1: mol.SetProp('_Name', str(i))
            # self.inactive_names.append(mol.GetProp('_Name')
            i += 1
        print('Actives', len(self.active_names))
        print('Inactives', len(self.inactive_names))
        return ()

    def process_bind(self):

        active_output, inactive_output, n_actives, n_inactives = self.split_sdf()
        if not self.debug:
            output_active_proc = pr.ligprep(active_output, output="active_processed.mae")
            output_active_proc = ft.mae_to_sd(output_active_proc,
                                              output=os.path.join(self.folder_output, 'actives.sdf'))
            output_inactive_proc = pr.ligprep(inactive_output, output="inactive_processed.mae")
            output_inactive_proc = ft.mae_to_sd(output_inactive_proc,
                                                output=os.path.join(self.folder_output, 'inactives.sdf'))
        else:
            output_active_proc = active_output
            output_inactive_proc = inactive_output

        return output_active_proc, output_inactive_proc


def parse_args(parser):
    parser.add_argument("--bindingdb", type=str, help='Location of file( e.g. /home/moruiz/cyp/bindingdb/cyp2c9.sdf')
    parser.add_argument("--mol_to_read", type=int, help="Number of molecules to read from pubchem (e.g. 100)")


if __name__ == '__main__':
    bindingdb = '/home/moruiz/cyp/bindingdb/cyp2c9.sdf'
    bindb = BindingDB(bindingdb, train=True, test=False, folder_to_get='.', n_molecules_to_read=None, folder_output='.',
                      production=False, debug=False)
    bindb.process_bind()
