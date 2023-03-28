import krippendorff
import numpy as np

def print_units_and_annotators(df):
    """"
    Print the number of units and annotators in a dataframe.
    :param df: a dataframe with columns 'pair_id' and 'annotator_id'"""
    unit_n = len(df['pair_id'].unique())
    annotator_n = len(df['annotator_id'].unique())
    print(f'Number of units: {unit_n}')
    print(f'Number of annotators: {annotator_n}')


def calculate_krippendorff_alpha(df):
    """Calculate Krippendorff's alpha for the given annotations dataframe"""
    def convert_to_ratings(df):
        unit_n = len(df['pair_id'].unique())
        annotator_n = len(df['annotator_id'].unique())
        ratings = np.full((annotator_n, unit_n), np.nan)
        for i, row in df.iterrows():
            ratings[row['annotator_idx'], row['unit_idx']] = row['sim_rating']
        return ratings

    annotations_df = df.copy()

    annotator2idx = {annotator: idx for idx, annotator in enumerate(annotations_df['annotator_id'].unique())}
    annotations_df['annotator_idx'] = annotations_df['annotator_id'].map(annotator2idx)

    unit2idx = {unit: idx for idx, unit in enumerate(annotations_df['pair_id'].unique())}
    annotations_df['unit_idx'] = annotations_df['pair_id'].map(unit2idx)
        
    ratings = convert_to_ratings(annotations_df)
    print(krippendorff.alpha(ratings, level_of_measurement='ordinal'))
    return annotations_df, ratings