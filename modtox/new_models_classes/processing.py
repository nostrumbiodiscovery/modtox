from col import Collection
import itertools

def main(actives_sdf, inactives_sdf, glide_features_csv):
    col = Collection(actives_sdf, inactives_sdf)
    col.add_glide_features(glide_features_csv)
    col.calculate_topo()
    col.calculate_mordred()
    col.balance()
    
    combinations = [ comb for comb in itertools.product((True, False), repeat=3) ]
    del combinations[-1]

    df_dict = dict()
    for comb in combinations:
        df = col.to_dataframe(glide=comb[0], mordred=comb[1], topo=comb[2])
        name = get_df_name(comb[0], comb[1], comb[2])
        df_dict[name] = df 
    return df_dict

def get_df_name(glide, mordred, topo):
    name = list()
    if glide:
        name.append("glide")
    if mordred:
        name.append("mordred")
    if topo:
        name.append("topo")
    return "_".join(name)
    