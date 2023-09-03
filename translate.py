from tqdm import trange

import dotenv
import os

import numpy as np
import pandas as pd
import requests

dotenv.load_dotenv()

def deepl_translate(texts):
    resp = requests.post(
        'https://api-free.deepl.com/v2/translate',
        headers={'Authorization': f'DeepL-Auth-Key {os.getenv("DEEPL_KEY")}'},
        json={'text': texts, 'target_lang': 'EN'}
    )
    return [item['text'] for item in resp.json()['translations']]

if __name__ == "__main__":
    batch_size = 50
    df_fr = pd.read_json('ML-ESG-2_French_Train.json')
    n_iters = df_fr.shape[0] // batch_size + 1

    translated_texts = []
    for i in trange(n_iters):
        try:
            results = deepl_translate(df_fr['news_content'].iloc[i*batch_size:(i+1)*batch_size].values.tolist())
            translated_texts.append(results)
        except:
            print(f"Error at batch {i}")
    
    df = pd.DataFrame({
        'news_content': np.concatenate(translated_texts),
        'impact_type': df_fr['impact_type']
    })

    df.to_csv('fr_en.csv', index=False)