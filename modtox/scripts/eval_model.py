
# import os
# import itertools
# from dataclasses import dataclass
# import pandas as pd

# DATA = os.path.join(os.path.dirname(__file__), "data")
# ACTIVES = os.path.join(DATA, "actives.sdf")
# INACTIVES = os.path.join(DATA, "inactives.sdf")
# GLIDE = os.path.join(DATA, "glide_features.csv")
# DF = os.path.join(DATA, "glide_mordred_topo.csv")

# @dataclass
# class PreprocessCombination:
#     X_train: pd.DataFrame
#     y_train: pd.DataFrame
#     X_test: pd.DataFrame
#     y_test: pd.DataFrame

# def main():
#     df = pd.read_csv(DF, index_col=0)
#     ds = DataSet(df)
#     options = {
#             "outliers": [0, 1, 10, 100],
#             "balance": ["undersampling", "oversampling"],
#             "scaling": ["standarize", "normalize"],
#     }
#     combinations = list(itertools.product(*list(options.values())))
#     datasets = dict()
#     for comb in combinations:
#         X_train, y_train, X_test, y_test = ds.prepare(comb[0], comb[1], comb[2])
#         to_str = [str(x) for x in comb]
#         datasets["_".join(to_str)] = PreprocessCombination(X_train, y_train, X_test, y_test)

#     models = dict()
#     for name, ds in datasets.items():
#         m = Model(ds.X_train, ds.y_train, ds.X_test, ds.y_test)
#         models[name] = m.build_model()
    
#     records = list()
#     for name, summ in models.items():
#         outlier_threshold, balance_method, scaling = name.split(sep="_")
#         records.append([
#             outlier_threshold,
#             balance_method,
#             scaling,
#             summ.step,
#             summ.selector_scores,
#             summ.selected_features,
#             summ.test_score,
#             summ.best_params,
#             summ.matrix,
#             summ.accuracy,
#             summ.precision,
#         ])
#     df = pd.DataFrame(records, columns=[
#             "outlier_threshold",
#             "balance_method",
#             "scaling",
#             "feat_selection_step",
#             "selector_scores",
#             "selected_features",
#             "test_score",
#             "best_params",
#             "matrix",
#             "accuracy",
#             "precision",
#     ])

#     df.to_csv("overview.csv")

# if __name__ == "__main__":
#     main()