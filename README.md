# Bi-Lingual Persona Stability

Cross-lingual IPIP-50 personality evaluation for small open language models in
English and Burmese.

The project contains:

- `llm_pipeline.py`: stateless Ollama inference pipeline.
- `analyze_results.py`: Big Five scoring, parse-failure reporting, and chart generation.
- `ipip-50_english.json` and `ipip-50_burmese.json`: parallel questionnaire data.
- `llm_persona_results.csv`: raw model responses.
- `llm_persona_summary.csv`: model-language-trait summary scores.
