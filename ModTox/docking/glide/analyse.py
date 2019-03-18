import argparse
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from argparse import RawTextHelpFormatter
import os
import subprocess
import ModTox.constants.constants as cs


def analyze(glide_files, model=True, max=False, csv=[]):
    glide_results = []
    if model:
        for i, glide_file in enumerate(glide_files):
        	results_merge = merge([glide_file], output="results_merge_{}.mae".format(i))
        	results_mae = sort_by_dock_score([results_merge,], output="data_{}.txt".format(i))
        	glide_results.append(to_dataframe(results_mae, output="results_{}.csv".format(i), iteration=i))
        all_results = join_results(glide_results)
        print(all_results)
    elif max:
        results_merge = merge(glide_files,  output="results_merge.mae")
        output = best_poses(results_merge)
    elif csv:
	csv, active, inactive = csv
        output, n_active = add_activity_feature(csv, active, inactive)
        summeryze_results(output, n_active)

def sort_by_dock_score(glide_files, schr=cs.SCHR, output="data.txt"):
    glide_sort_bin = os.path.join(schr, "utilities/glide_sort")
    command = "{} -r {} {}".format(glide_sort_bin , output, " ".join(glide_files))
    print(command)
    subprocess.call(command.split())
    return output
        
def merge(glide_files, schr=cs.SCHR, output="results_merge.mae"):
    glide_merge_bin = os.path.join(schr, "utilities/glide_merge")
    command = "{} {} -o {} ".format(glide_merge_bin , " ".join(glide_files), output)
    print(command)
    subprocess.call(command.split())
    return output
    
def best_poses(glide_file,  schr=cs.SCHR, output="final_resuts.mae"):
    schr_bin =  os.path.join(schr, "run")
    glide_summary_bin = os.path.join(schr, "mmshare-v4.2/python/common/glide_blocksort.py")
    command = "{} {} {} {} ".format(schr_bin, glide_summary_bin, glide_file,  output)
    print(command)
    subprocess.call(command.split())

def to_dataframe(glide_results_file, output="results.csv", write=True, iteration=None):
    found=False
    info = []
    with open(glide_results_file, "r") as f:
	for line in f:
            if line.split():
	        if line.startswith("Rank"):
                    columns = line.split()
                    n_columns = len(columns)
	            headers = columns[: n_columns-1]
	            found = True
		    if iteration:
			headers = ["{}_receptor_{}".format(header, iteration) if header != "Title" else header for header in headers]
                elif found and line.split()[0].isdigit():
                    ligand_info = line.split()
                    n_fields = len(ligand_info)
 		    if n_fields == n_columns:
		        info.append(ligand_info[: n_fields-1])
    
    df = pd.DataFrame(info, columns=headers)
    if write:
        df.to_csv(output)
    return df

def add_activity_feature(csv, active, inactive, output="csv_activity.csv"):
    assert active.split(".")[-1] == "sdf", "active file must be sdf"
    assert inactive.split(".")[-1] == "sdf", "active file must be sdf"

    actives = Chem.SDMolSupplier(active)
    inactives = Chem.SDMolSupplier(inactive)
    print("Active, Inactive")
    print(len(actives), len(inactives))

    actives_titles = [ mol.GetProp("_Name") for mol in actives if mol]
    inactives_titles = [ mol.GetProp("_Name") for mol in inactives if mol]

    new_lines = []
    with open(csv, "r") as f:
	for i, line in enumerate(f):
            if i == 0:
                new_lines.append(line.strip("\n") + ',"active"\n')
            else:
                title = line.split(",")[0].strip('"')
                if title in actives_titles:
                    new_lines.append(line.strip("\n") +",1\n")
                elif title in inactives_titles:
                    new_lines.append(line.strip("\n") +",0\n")
                else:
                    new_lines.append(line)

    with open(output, "w") as f:
        f.write("".join(new_lines))

    return output, len(actives)            
            
def summeryze_results(csv, tresh):

    with open(csv, "r") as f:

        false_positives = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0 

        lines = [ line.strip("\n") for line in f.readlines()[1:] ]

	for i, line in enumerate(lines):
            try:
                if i < tresh:
                    if int(line.split(",")[-1]) == 1:
                        true_positive += 1
                    else:
                        false_positives += 1
                else:
                    if int(line.split(",")[-1]) == 0:
                        true_negative += 1
                    else:
                        false_negative += 1
            except ValueError:
                pass

    print("TP {}, FP {}, TN {}, FN {}".format(true_positive, false_positives, true_negative, false_negative))
    return true_positive, false_positives, true_negative, false_negative
        
               
         

def join_results(files, output="glide_features.csv"):
    for i, glide_file in enumerate(files):
        try:
            if i == 0:
                df = pd.merge(files[i], files[i+1], on="Title", how="outer")
            else:
	        df = pd.merge(df, files[i+1], on="Title", how="outer")
        except IndexError:
            df.to_csv(output)
            return df


def conf(TP, FP, TN, FN):
    df_cm = pd.DataFrame([[TP, FP], [FN,TN]], index = [i for i in "PN"],
                      columns = [i for i in "PN"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze glide docking\n  \
    i.e python -m ModTox.docking.glide.analyze glide_file1 glide_file2", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--glide_files",  nargs="+", help='Glide files to be analyze')
    parser.add_argument("--csv",  nargs="+", help='Csv to be analyze')
    args = parser.parse_args()
    return args.glide_files, args.csv

if __name__ == "__main__":
    glide_files, csv = parse_args()
    analyze(glide_files, model=True, max=False, csv=csv)
    #conf(116, 46, 345, 47)
