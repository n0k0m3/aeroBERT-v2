# aeroBERTv2

## Requirements

```bash
conda env create -f environment.yml
conda activate aeroBERTv2
```

## Build your own up-to-date dataset

Download `title-14.xml` from CFR to `data/`

### Preprocessing

Generate train and test dataset for:
- Masked language modeling/pretraining and perplexity evaluation (from Title 14 CFR)
- aeroBERT-NER (from archanatikayatray/aeroBERT-NER)
- aeroBERT-classification (from archanatikayatray/aeroBERT-classification)

```bash
python data/pretrain_gen.py
python data/ner_gen.py
python data/classifier_gen.py
```