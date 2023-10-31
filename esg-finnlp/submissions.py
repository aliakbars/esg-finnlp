import pandas as pd
import json

if __name__=='__main__':
    # all_train_df = pd.read_json(
    #     'esg-finnlp/data/raw/ML-ESG-2_English_Train.json'
    # )
    # all_train_df = all_train_df.rename(
    #     columns={
    #         'news_content': 'sentence',
    #         'impact_type': 'label'
    #     }
    # )[[
    #     'sentence',
    #     'label'
    # ]]
    # all_train_df.to_json(
    #     'esg-finnlp/data/raw/train_all.json', orient='records'
    # )

    # rows = []
    # with open('esg-finnlp/data/submissions/roberta-large/fr_test_label_w_prob.json', 'r') as fr:
    #     sub_en_roberta = json.load(fr)
    #     for row in sub_en_roberta:
    #         new_row = row.copy()
    #         if row['impact_type'][0]['label'] == 'LABEL_0':
    #             new_row['impact_type'] = 'Opportunity'
    #         elif row['impact_type'][0]['label'] == 'LABEL_1':
    #             new_row['impact_type'] = 'Risk'
    #         else:
    #             raise ValueError
    #         rows.append(new_row)
    
    # with open('esg-finnlp/data/submissions/roberta-large/AnakItik_French_2.json', 'w') as fw:
    #     json.dump(rows, fw)

    combined = []
    count1 = {'Opportunity': 0, 'Risk':0}
    count2 = {'Opportunity': 0, 'Risk':0}
    count_diff = 0
    rows = []
    rows_t5 = []
    with open('esg-finnlp/data/submissions/T5/AnakItik_French_1_raw.json', 'r') as fw1, \
        open('esg-finnlp/data/submissions/xlm-roberta-large/AnakItik_French_2.json', 'r') as fw2:
        json1 = json.load(fw1)
        json2 = json.load(fw2)
        for i in range(len(json1)):
            row1 = json1[i]
            row2 = json2[i]
            row_comb = row2.copy()
            row_t5 = row2.copy()
            row_t5['impact_type'] = row1['impact_type']
            count1[row1['impact_type']] += 1
            count2[row2['impact_type']] += 1
            if (row1['impact_type'] != row2['impact_type']):
                count_diff +=1
                print(row1['news_content'])
                print(row1['impact_type'])
                print(row2['impact_type'])
                row_comb['impact_type'] = 'Cannot Distinguish'
            rows.append(row_comb)
            rows_t5.append(row_t5)
    print(count_diff)
    print(count1)
    print(count2)
    
    with open('esg-finnlp/data/submissions/mix/AnakItik_French_3.json', 'w') as fm:
        json.dump(rows, fm)
    
    with open('esg-finnlp/data/submissions/T5/AnakItik_French_1.json', 'w') as fm:
        json.dump(rows_t5, fm)

    # with open('esg-finnlp/data/submissions/T5/AnakItik_English_1_raw.json', 'r') as fw1, \
    #     open('esg-finnlp/data/submissions/T5/AnakItik_English_1.json', 'r') as fw2:
    #     json1 = json.load(fw1)
    #     json2 = json.load(fw2)
    #     for i in range(len(json1)):
    #         row1 = json1[i]
    #         row2 = json2[i]
    #         if row1['impact_type'] != row2['impact_type']:
    #             print(row1)
    
    # with open('esg-finnlp/data/submissions/T5/AnakItik_English_1_raw.json', 'r') as fw1, \
    #     open('esg-finnlp/data/submissions/roberta-large/AnakItik_English_2.json', 'r') as fw2, \
    #     open('esg-finnlp/data/submissions/roberta-large/AnakItik_English_3.json', 'r') as fw3:
    #     json1 = json.load(fw1)
    #     json2 = json.load(fw2)
    #     for i in range(len(json1)):
    #         row1 = json1[i]
    #         row2 = json2[i]