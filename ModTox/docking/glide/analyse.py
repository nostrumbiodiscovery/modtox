import argparse
import seaborn as sn
import matplotlib.pyplot as plt
from itertools import tee
import pandas as pd
from rdkit import Chem
from argparse import RawTextHelpFormatter
import os
import subprocess
import ModTox.constants.constants as cs
from itertools import chain


def analyze(glide_files, active=False, inactive=False, model=True, best=False, csv=[]):
    glide_results = []
    if model:
        for i, glide_file in enumerate(glide_files):
        	results_merge = merge([glide_file], output="results_merge_{}.mae".format(i))
        	results_mae = sort_by_dock_score([results_merge,], output="data_{}.txt".format(i))
        	glide_results.append(to_dataframe(results_mae, output="results_{}.csv".format(i), iteration=i))
        all_results = join_results(glide_results)
        print(all_results)
    elif best:
        results_merge = merge(glide_files,  output="results_merge.mae")
        best_poses_mae = best_poses(results_merge)
        best_poses_csv = csv_report(best_poses_mae)
        if active and inactive:
            #Sometimes some molecules are lost along the way for rdkit problems
            # that's the n_initial _active variable
            output, n_active, n_initial_active, n_initial_inactive = add_activity_feature(best_poses_csv, active, inactive)
            TP, FP, TN, FN = summeryze_results(output, n_active, n_initial_active, n_initial_inactive)
            conf(TP, FP, TN, FN)
            

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
    return output


def csv_report(glide_file,  schr=cs.SCHR, properties = ["s_m_title", "r_i_docking_score"],  output="results_all_properties.csv"):
    csv_report_bin =  os.path.join(schr, "utilities/proplister")
    properties = [ "-p {}".format(prop) for prop in properties]
    command = "{} -a -c {} {} > {} ".format(csv_report_bin, " ".join(properties), glide_file,  output)
    print(command)
    os.system(command)
    return output


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
    
    actives = iter(())
    for sdf_file in active:
        assert sdf_file.split(".")[-1] == "sdf", "active file must be sdf"
        actives = chain(actives, Chem.SDMolSupplier(sdf_file))
    inactives = iter(())
    for sdf_file in inactive:
        assert sdf_file.split(".")[-1] == "sdf", "active file must be sdf"
        inactives = chain(actives, Chem.SDMolSupplier(sdf_file))
 
    actives, actives_copy = tee(actives)
    inactives, inactives_copy = tee(inactives)


    actives_titles = [ mol.GetProp("_Name") for mol in actives if mol]
    inactives_titles = [ mol.GetProp("_Name") for mol in inactives if mol]

    print("Active, Inactive")
    print(len(actives_titles), len(inactives_titles))

    new_lines = []
    with open(csv, "r") as f:
	for i, line in enumerate(f):
            if i == 0:
                new_lines.append(line.strip("\n") + ',"active"\n')
            else:
                title = line.split(",")[0].strip('"').strip("'")
                if title in actives_titles:
                    new_lines.append(line.strip("\n") +",1\n")
                elif title in inactives_titles:
                    new_lines.append(line.strip("\n") +",0\n")
                else:
                    new_lines.append(line)

    with open(output, "w") as f:
        f.write("".join(new_lines))

    n_real_active = sum(1 for _ in actives_copy)
    n_real_inactvie = sum(1 for _ in inactives_copy)

    return output, len(actives_titles), n_real_active, n_real_inactvie 
            
def summeryze_results(csv, tresh, n_active, n_inactive):

    with open(csv, "r") as f:

        false_positive = 0
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
                        false_positive += 1
                else:
                    if int(line.split(",")[-1]) == 0:
                        true_negative += 1
                    else:
                        false_negative += 1
            except ValueError:
                pass
    false_negative += n_active - true_positive -  false_negative
    true_negative += n_inactive - true_negative -  false_positive

    print("TP {}, FP {}, TN {}, FN {}".format(true_positive, false_positive, true_negative, false_negative))
    return true_positive, false_positive, true_negative, false_negative
        
               
         

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


def conf(TP, FP, TN, FN, output="confusion_matrix.png"):
    df_cm = pd.DataFrame([[TP, FP], [FN,TN]], index = [i for i in "PN"],
                      columns = [i for i in "PN"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig(output)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze glide docking\n  \
    i.e python -m ModTox.docking.glide.analyze glide_file1 glide_file2", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--glide_files",  nargs="+", help='Glide files to be analyze')
    parser.add_argument("--csv",  nargs="+", help='Csv to be analyze')
    parser.add_argument("--model",  action="store_true", help='Create model matrix from docking results')
    parser.add_argument("--best",  action="store_true", help='Retrieve best poses from docking results')
    parser.add_argument("--act",  nargs="+", help='Files with all active structures used for docking. Must be a sdf file', default=None)
    parser.add_argument("--inact",  nargs="+", help='Files with all inactive structures used for docking. Must be a sdf file', default=None)
    args = parser.parse_args()
    return args.glide_files, args.csv, args.model, args.best, args.act, args.inact

if __name__ == "__main__":
    glide_files, csv, model, best, act, inact = parse_args()
    analyze(glide_files, model=model, best=best, csv=csv, active=act, inactive=inact)
    #conf(116, 46, 345, 47)
