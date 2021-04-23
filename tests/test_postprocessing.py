import glob
import os

import tests.config as tc

TMP = os.path.join(os.path.dirname(__file__), "tmp")


def test_confusion_matrix(output="confusion_matrix.png"):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(folder=TMP)
    pp.conf_matrix(output_conf=output)
    assert os.path.exists(os.path.join(TMP, output))


def test_uncertainties():
    pp = tc.retrieve_processor(train_data=True, clf_stacked=True, folder=TMP)
    pp.calculate_uncertanties()


def test_umap(output_umap="umap_proj"):
    tc.clean(glob.glob(os.path.join(TMP, output_umap, "*")))
    pp = tc.retrieve_processor(folder=TMP, train_data=True)
    pp.UMAP_plot(output_umap=output_umap, single=True, wrong=True, wrongall=True, traintest=True, wrongsingle=True)
    assert any(os.path.join(TMP, output_umap, "*"))


def test_pca(output_pca="pca.png"):
    tc.clean(os.path.join(TMP, output_pca))
    pp = tc.retrieve_processor(folder=TMP)
    pp.PCA_plot(output_pca=output_pca)
    assert os.path.exists(os.path.join(TMP, output_pca))


def test_tsne(output_tsne="tsne.png"):
    tc.clean(os.path.join(TMP, output_tsne))
    pp = tc.retrieve_processor(folder=TMP)
    pp.tsne_plot(output_tsne=output_tsne)
    assert os.path.exists(os.path.join(TMP, output_tsne))


def test_ROC(output="ROC.png"):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(folder=TMP)
    pp.ROC(output_ROC=output)
    assert os.path.exists(os.path.join(TMP, output))


def test_PR(output="PR.png"):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(folder=TMP)
    pp.PR(output_PR=output)
    assert os.path.exists(os.path.join(TMP, output))


def test_shap(output='shap.png'):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(train_data=True, folder=TMP)
    pp.shap_values(output_shap=output, debug=True)
    assert os.path.exists(os.path.join(TMP, output))


def test_distributions(output='distributions'):
    tc.clean(glob.glob(output + "*"))
    pp = tc.retrieve_processor(train_data=True, folder=TMP)
    pp.distributions(output_distributions=output)
    assert any(glob.glob(os.path.join(TMP, 'distributions_*')))


def test_feature_importance(output='feature_importance.txt'):
    tc.clean(os.path.join(TMP, output))
    pp = tc.retrieve_processor(train_data=True, folder=TMP)
    pp.feature_importance(output_features=output)
    assert os.path.exists(os.path.join(TMP, output))


def test_domain_analysis(output_densities="thresholds_vs_density.png", output_thresholds="threshold_analysis.txt",
                         output_distplots="distplot"):
    output_densities = os.path.join(TMP, output_densities)
    output_thresholds = os.path.join(TMP, output_thresholds)
    output_distplots = os.path.join(TMP, output_distplots)
    output = [output_densities, output_thresholds] + glob.glob(output_distplots + "*")
    tc.clean(output)
    pp = tc.retrieve_processor(train_data=True)
    pp.domain_analysis(output_densities=output_densities, output_thresholds=output_thresholds,
                       output_distplots=output_distplots, debug=True)
    assert any(output)
