import pandas as pd
import modtox.data.dude as dd


class GPCRDB(dd.DUDE):


    def __init__(self, csv):
        self.csv = csv

    def Filter(self, activity_column="Activity Value", columns=["Assay Description"], strings=["125I"], output_csv="filtered.csv"):
        #Column and string must be lists [Column1] [String in column 1]
        for column, string in zip(columns, strings):
            df = pd.DataFrame.from_csv(self.csv)
            df_filter = df[df[column].str.contains(string)]
            df_final = df_filter[activity_column]
            df_final.to_csv(output_csv)
        return output_csv

    def chembl_names_from_csv(self, csv):
        df = pd.DataFrame.from_csv(csv)
        return df.index
        


def add_args():
    parser.add_argument('--gpcr_csv', type=str,
                        help='csv with informtion of gpcr ligands activities')
    parser.add_argument('--gpcr_activity', type=str,
                        help='Column name of the activities column', default="Activity Value")
    parser.add_argument('--gpcr_columns', nargs="+",
                        help='columns to filter', default=["Assay Description"])
    parser.add_argument('--gpcr_filters', nargs="+",
                        help='strings to look at each column', default=["125I"])
    parser.add_argument('--gpcr_output', type=str,
                        help='output filtered file', default="filtered.csv")


def process_gpcrdb(gpcr_csv, gpcr_activity="Activity Value", gpcr_columns=["Assay Description"], gpcr_filters=["125I"], gpcr_output="filtered.csv"):
    gpcr_obj = GPCRDB(gpcr_csv)
    output = gpcr_obj.Filter(activity_column=gpcr_activity, columns=gpcr_columns, strings=gpcr_filters, output_csv=gpcr_output)
    chmbl_names = gpcr_obj.chembl_names_from_csv(output)
    return gpcr_obj.to_sdf(chmbl_names)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build 2D QSAR model')
    parse_args(parser)
    args = parser.parse_args()
    process_gpcrdb(args.gpcr_csv, activity_column=args.gpcr_activity, column=args.gpcr_columns, string=args.gpcr_filters, output_csv=args.gpcr_output)
    
