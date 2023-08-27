# TASK PAGE

https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp2023/shared-task-esg-impact

# IMPORTANT DATES

Time zone: Anywhere On Earth (AOE)

1. Registration Open: July 18, 2023
2. Training set release (Chinese): July 31, 2023
3. Training set release (English & French): Aug. 21, 2023
4. Test set release: Sep 20, 2023
5. System's outputs submission deadline: Sep 25, 2023
6. Release of results: Sep 28, 2023
7. Shared task paper submissions due: Oct 3, 2023 - Shared Task Paper Submission System: Available soon
8. Notification: Oct 5, 2023
9. Camera-Ready Version of Shared Task Paper Due: Oct 8, 2023

# TODOS

1. Dataset enrichment
    a. Translate English, French, Chinese, Japanese articles to English and French
    b. Data augmentation (GPT3Mix? https://arxiv.org/abs/2104.08826)
2. Encoders:
    a. Try the typical encoders: DeBERTa, RoBERTa, FinBERT, CamemBERT/Fr (base/large).
    b. Prepare ensembling
3. Prompt Engineering:
    a. Try the newest LLMs in HF: llama, falcon, bloom
    b. Explore prompts, chain-of-thought, self-criticism

# REFERENCES

1. Previous winner: https://arxiv.org/abs/2306.06662, github: https://github.com/finMU/ML-ESG_codes.
    - English best performer: roberta-base-mix (large) trained on augmented data + hard-voting ensemble
    - French best performer: mdeberta-mix (large)
2. Chinese dataset: 
    - https://github.com/ymntseng/FinNLP2023_ML-ESG-2_ChineseCrawler