from collections import defaultdict
import argparse
import os

import pandas as pd
from tqdm.auto import tqdm


def get_chunk_features(train_chunk: pd.DataFrame, attr_chunk: pd.DataFrame) -> pd.DataFrame:
    # Count nan
    attr_chunk['nan_cnt'] = attr_chunk.isna().sum(axis=1)
    
    # Get friends
    friends = defaultdict(set)
    for u, subset in train_chunk.groupby('u'):
        friends[u] = set(subset['v'])
    for v, subset in train_chunk.groupby('v'):
        friends[v] = friends[v].union(set(subset['u']))
    
    attr_chunk['friend_cnt'] = attr_chunk.apply(
        lambda row: len(friends[row['u']]),
        axis=1
    )
    
    # Count common friends
    train_chunk['common_friends_cnt'] = train_chunk.apply(
        lambda row: len(friends[row['u']].intersection(friends[row['v']])),
        axis=1
    )
    
    # Merge dataframes
    train_chunk = pd.merge(
        train_chunk,
        attr_chunk,
        how='left',
        on=['ego_id', 'u']
    )
    train_chunk = pd.merge(
        train_chunk,
        attr_chunk.rename(columns={'u': 'v'}),
        how='left',
        on=['ego_id', 'v']
    )
    
    # Check if city, school or university are same
    train_chunk['same_city_id'] =\
        (train_chunk['city_id_x'] == train_chunk['city_id_y']) & (train_chunk['city_id_x'].notna())
    train_chunk['same_school'] =\
        (train_chunk['school_x'] == train_chunk['school_y']) & (train_chunk['school_x'].notna())
    train_chunk['same_university'] =\
        (train_chunk['university_x'] == train_chunk['university_y']) & (train_chunk['university_x'].notna())
    
    # Generate dummy features for sex
    train_chunk = train_chunk.join(
        pd.get_dummies(train_chunk['sex_x'], dummy_na=True, prefix='sex_x')
    ).join(
        pd.get_dummies(train_chunk['sex_y'], dummy_na=True, prefix='sex_y')
    )

    # Drop unnecessary features
    train_chunk = train_chunk.drop([
        'city_id_x',
        'city_id_y',
        'sex_x',
        'sex_y',
        'school_x',
        'school_y',
        'university_x',
        'university_y',
    ], axis=1)
    
    return train_chunk


def get_features(train: pd.DataFrame, attr: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists('chunks'):
        os.mkdir('chunks')

    for ego_id in tqdm(train['ego_id'].unique()):
        # Get dataset chunk
        train_chunk = train[train['ego_id'] == ego_id]
        attr_chunk = attr[attr['ego_id'] == ego_id]
        
        # Compute features
        train_chunk = get_chunk_features(train_chunk, attr_chunk)
        
        # Save chunk
        train_chunk.\
            to_csv(f'chunks/ego_id_{ego_id}.csv')
    
    # Merge files
    file_paths = [
        'chunks/' + file_name
        for file_name in os.listdir('chunks')
    ]

    train_merged = pd.concat(
        map(pd.read_csv, file_paths),
        ignore_index=True,
    )
    return train_merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature generation script')
    parser.add_argument('-a', dest='attr', required=True, type=str)
    parser.add_argument('-i', dest='input', required=True, type=str)
    parser.add_argument('-o', dest='output', required=True, type=str)
    args = parser.parse_args()

    attr = pd.read_csv(args.attr)
    input_df = pd.read_csv(args.input)

    output_df = get_features(input_df, attr)
    output_df.to_csv(args.output)
