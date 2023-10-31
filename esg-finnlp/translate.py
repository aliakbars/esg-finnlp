from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import trange

import click
import dotenv
import os

import numpy as np
import pandas as pd
import requests

dotenv.load_dotenv()

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def translate(texts):
    text_batch = [f"Translate from French to English: {text}" for text in texts]
    inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=1024)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def deepl_translate(texts):
    resp = requests.post(
        'https://api-free.deepl.com/v2/translate',
        headers={'Authorization': f'DeepL-Auth-Key {os.getenv("DEEPL_KEY")}'},
        json={'text': texts, 'target_lang': 'EN'}
    )
    return [item['text'] for item in resp.json()['translations']]

@click.command
@click.option('-a', '--api', is_flag=True)
@click.option('--batch_size', default=50)
def main(api, batch_size):
    print("Loading data...")
    df_fr = pd.read_json('ML-ESG-2_French_Train.json')
    n_iters = df_fr.shape[0] // batch_size + 1

    translated_texts = []
    for i in trange(n_iters):
        try:
            texts = df_fr['news_content'].iloc[i*batch_size:(i+1)*batch_size].values.tolist()

            if api:
                results = deepl_translate(texts)
            else:
                results = translate(texts)

            translated_texts.append(results)
        except:
            print(f"Error at batch {i}")
    
    df = pd.DataFrame({
        'news_content': np.concatenate(translated_texts),
        'impact_type': df_fr['impact_type']
    })

    df.to_csv('fr_en.csv', index=False)

if __name__ == "__main__":
    main()