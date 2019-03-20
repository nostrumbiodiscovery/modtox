import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
from sklearn import svm
import sys
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ModTox.ML.descriptors_2D_ligand import *


TITLE_MOL = "molecules"
LABELS = "labels"


class GenericModel():

    def __init__(self, active, inactive, clf):
        self.active = active
        self.inactive = inactive
        self.clf = clf
        self.data = self._load()
        self.features = self.data.iloc[:, :-1]
        self.labels = self.data.iloc[:, -1]

    
    def _load(self):
        """
        Separate between train and test dataframe
    
        Input:
            :input_file: str
            :sdf_property: str
        Output:
            :xtrain: Pandas DataFrame with molecules for training
            :xtest: Pandas DataFrame with molecules for testing
            :ytrain: Pandas Dataframe with labels for training
            :ytest: Pandas DataFrame with labels for testing
        """
    
        
        actives = [ mol for mol in Chem.SDMolSupplier(self.active) if mol ]
        inactives = [ mol for mol in Chem.SDMolSupplier(self.inactive) if mol ]
        titles = [ mol.GetProp("_Name") for mol in actives ] 
        
        actives_df = pd.DataFrame({TITLE_MOL: actives })
        inactives_df =  pd.DataFrame({TITLE_MOL: inactives })
    
        actives_df[LABELS] = [1,] * actives_df.shape[0]
        inactives_df[LABELS] = [0,] * inactives_df.shape[0]
    
        molecules = pd.concat([actives_df, inactives_df])
    
        print("Active, Inactive")
        print(actives_df.shape[0], inactives_df.shape[0])

        self.data = molecules
    
        return self.data

    def GENERAL_ASPECTS(self):
        molecular_data = [ TITLE_MOL, ]
        numeric_features = ['fingerprint', 'fingerprintMACS', 'descriptors']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler())
                ])
        
        molec_transformer = FeatureUnion([
            ('fingerprintMACS',Fingerprints_MACS()),
            ('fingerprint', Fingerprints()),
            ('descriptors', Descriptors()),
                    ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                    ('mol', molec_transformer, molecular_data)])
        
        pre = Pipeline(steps=[('transformer', preprocessor),
                                  ('scaler', numeric_transformer),
                                                       ])
        return pre
            
        

    def fit_transform(self, x_train, y_train, output="model.txt", load=False):
        
    
        if not load:
            pre = self.GENERAL_ASPECTS() 
            x_train_trans = pre.fit_transform(x_train)
            np.savetxt(output, x_train_trans) # Save Dataset
    
        else:
            x_train_trans = np.loadtxt(output) # Load Dataset
    
        np.random.seed(7)
    
        prediction = cross_val_predict(self.clf, x_train_trans, y_train, cv=100)
    
        print(confusion_matrix(y_train, prediction))


if __name__ == "__main__":
    clf = svm.SVC(C=1, gamma=1, kernel="linear")
    model = GenericModel(sys.argv[1], sys.argv[2], clf)
    model.fit_transform(model.features, model.labels, load=True)
