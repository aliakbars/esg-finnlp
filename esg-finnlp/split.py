import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

if __name__=='__main__':
    # French split
    # generate raw data with index, content only
    df = pd.read_csv(
        'esg-finnlp/data/translation/fr_en_gpt-3.5-turbo.csv'
    ).rename(
        columns={
            'news_content_en': 'sentence',
            'impact_type': 'label',
            'news_title': 'group'
        }
    ).dropna()

    # English split
    # df = pd.read_json(
    #     'esg-finnlp/data/raw/ML-ESG-2_English_Train.json'
    # ).rename(
    #     columns={
    #         'news_content' : 'sentence',
    #         'impact_type': 'label',
    #         'URL': 'group'
    #     }
    # )

    # Splitter
    x, y, group = df['sentence'], df['label'], df['group']

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1337)

    output = {}
    folds = splitter.split(x, y, group)
    for i, (train, test) in enumerate(folds):
        train_df, test_df = df.iloc[train][['sentence', 'label']], df.iloc[test][['sentence', 'label']]
        train_df.to_json(f'esg-finnlp/data/splits/fr/{i+1}_train.json', orient='records')
        test_df.to_json(f'esg-finnlp/data/splits/fr/{i+1}_test.json', orient='records')