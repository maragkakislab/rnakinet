import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from plot_helpers import setup_palette
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class UpperQuantileScaler(BaseEstimator, TransformerMixin):
    def __init__(self, quantile=0.75):
        self.quantile = quantile

    def fit(self, X, y=None):
        self.scale_ = np.quantile(X, self.quantile, axis=0)
        return self

    def transform(self, X, y=None):
        return X / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    
prefix_to_scaler = {
    'standardized':StandardScaler,
    'robust':RobustScaler,
    'quantile':UpperQuantileScaler,
    'minmax':MinMaxScaler,
}

def main(args):
    table = pd.read_csv(args.table, sep='\t')
    table = add_normalized_halflifes(table, args.halflife_column, prefix_to_scaler)
    table.to_csv(args.output, sep='\t')
    
def add_normalized_halflifes(df, col, prefix_to_scaler):
    for prefix, scaler in prefix_to_scaler.items():
        df[f'{prefix}_'+col] = scaler().fit_transform(df[col].to_numpy().reshape(-1,1))
    df['unnormalized_'+col] = df[col]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--table', type=str, required=True)
    parser.add_argument('--halflife-column', type=str, required=True)
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    