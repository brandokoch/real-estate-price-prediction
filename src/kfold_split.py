import pandas as pd 
from sklearn import model_selection
import config

if __name__=="__main__":

    df = pd.read_pickle(config.TRAINING_DATA)

    df['kfold']= -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=5)

    for fold, (train_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_,'kfold']=fold

    df.to_pickle(config.TRAINING_DATA_FOLDS)