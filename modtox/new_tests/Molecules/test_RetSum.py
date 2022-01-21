import pickle5
import os
import matplotlib.pyplot as plt
from modtox.modtox.Molecules.target_col import RetrievalSummary

DATA = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "tests_outputs"
)
pickle_col = os.path.join(DATA, "P23316.pickle")
with open(pickle_col, "rb") as f:
    col = pickle5.load(f)


rs = RetrievalSummary(col)


def test_call_from_col():
    col.summarize_retrieval()


def test_molecules_databases_venn():
    fig, axs = plt.subplots(1, 2)
    rs._molecules_databases_venn(axs[0])
    rs._std_types_piechart(axs[1])
    axs.show()
    return


def test_info():
    rs._activities_table("ax")


def test_tags():
    rs.activity_assessment_table()


def test_merge():
    fig = rs.merge_plots()
    return
