import pandas as pd
import argparse
import ModTox.ML.preprocess_data as prep

def parse_args():
    parser = argparse.ArgumentParser(description='Build 3D-QSAR model to asses toxicology')
    parser.add_argument('--csv', type=str, help='Load datset from csv')
    args = parser.parse_args()
    return args.csv

class AR_model():


    def __init__(self):
        pass


    def load_model(pickle):
        loaded_model = pickle.load(open(pickle, 'rb'))

    def save_model(picke):
        pickle.dump(model, open(pickle, 'wb'))




def dataset_from_csv(csv):
    df = pd.DataFrame.from_csv(csv)
    return df

if __name__ == "__main__":
    csv = parse_args()
    df = dataset_from_csv(csv)
    prep.preprocess(df)
