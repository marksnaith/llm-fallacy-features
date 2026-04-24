# LLM Fallacy Feature Representations

This repository contains the data, analysis code, and prompts associated 
with the paper:

> Snaith, M., Pruś, J., Ziembicki, D., Chudziak, J., and Koszowy, M. 
> *Examining Interpretable Feature-Based Representations of Logical 
> Fallacies in Large Language Models*. 
> (Under review)

## Contents

- `prompt/` — feature extraction prompt used to query each model
- `data/` — output CSV files containing feature vectors for each model
- `analysis.py` — analysis script reproducing all reported results
- `requirements.txt` — Python dependencies

## Reproducing the Results

### From the provided data
To reproduce the analysis directly from the provided feature vectors:

```bash
pip install -r requirements.txt
python analysis.py
```

### From scratch
To reproduce the full pipeline from the raw input data:

1. Obtain the ElecDeb60to20 corpus from 
   https://github.com/pierpaologoffredo/ElecDeb60to20
2. The input examples used in this paper are drawn from 
   `data/fallacy_second_version.csv` in that repository, filtered 
   to the Appeal to Fear and Popular Opinion subcategories
3. Use the prompt in `prompt/` to query each model with the fallacious 
   argument examples one-by-one, replacing the `[[text]]` placeholder with the text of the argument. Each model returns a JSON object containing 
   feature scores as specified in the prompt
4. Convert the JSON responses to CSV format. The `text`, `fallacy`, 
   `subcategory` and `id` fields are taken from the ElecDeb60to20 input 
   data; the feature columns are populated from the model's JSON response. 
   The expected schema is shown in `data/schema.csv`
5. Run `python analysis.py`

## Notes
- The full pipeline code is not included in this release as it depends on 
  a private library. All inputs, outputs, and analysis code are provided 
  for full reproducibility of the reported results.
- When running from scratch, models are prompted to return a valid JSON 
  object only. However, LLMs may occasionally return malformed output and 
  appropriate error handling should be implemented.
- Due to the nondeterministic nature of LLMs, the feature vectors obtained when running from scratch may differ from those used in the paper (and provided in this repository).