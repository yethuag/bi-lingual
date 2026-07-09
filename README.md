# Bi-Lingual Persona Stability

Cross-lingual IPIP-50 personality evaluation for small open language models in
English and Burmese.

The project contains:

- `llm_pipeline.py`: stateless Ollama inference pipeline.
- `analyze_results.py`: Big Five scoring, parse-failure reporting, and chart generation.
- `visualize_persona_results.py`: publication radar, drift, and parse-failure figures.
- `kaggle_one_model_eval_and_visualize.py`: Kaggle/Hugging Face runner for one target model at a time.
- `ipip-50_english.json` and `ipip-50_burmese.json`: parallel questionnaire data.
- `llm_persona_results.csv`: raw model responses.
- `llm_persona_summary.csv`: model-language-trait summary scores.

Set `TARGET_MODEL` when reusing the Kaggle runner for another model, for example:

```bash
TARGET_MODEL=Qwen/Qwen2.5-14B-Instruct python kaggle_one_model_eval_and_visualize.py
```
