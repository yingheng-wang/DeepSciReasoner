# Data

This directory is populated by

```bash
bash scripts/download_data.sh
```

which fetches the prepared benchmarks from
[huggingface.co/datasets/yingheng/DeepSciReasoner](https://huggingface.co/datasets/yingheng/DeepSciReasoner)
and unpacks each one to `DATA/<benchmark>/{manifest.csv,split.pkl,graphs/}`.
See the main README and the dataset card for the benchmark list, file format and sources.
