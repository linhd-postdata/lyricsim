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
    # Pivot the DataFrame to create a table where rows represent text pairs and columns represent annotators
    pivot_df = df.pivot_table(index="pair_id", columns="annotator_id", values="sim_rating", aggfunc="first")

    # # Replace missing values with NaN
    alpha_df = pivot_df.replace({None: float("nan")})

    # Calculate Krippendorff's alpha
    alpha = krippendorff.alpha(alpha_df.transpose(), level_of_measurement='ordinal')
    
    return alpha


