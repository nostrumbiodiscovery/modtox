


def preprocess(df):
    thresh = int(df.shape[1]*0.8)
    df_dropna = df.dropna(thresh=thresh)
    print("Initial shape, Final shape")
    print(df.shape, df_dropna.shape)
