# Ahu's Paimon Toolkit

This is a toolkit for SiriusAhu's FYP, *PAIMON* (Player-Aware Intelligent Monitoring and Operations Navigator).

## `vLLM` Benchmark

This toolkit provides a series of benchmark functions for `vLLM` server.

**Note**: The `vLLM` benchmark needs the download of `QWEN3-VL-2B-Instruct` model, which could be automatically downloaded by running benchmark functions.

To speed up the download process, China users can set environment variable `HF_ENDPOINT` to `https://hf-mirror.com` before running benchmark functions.

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Usage

```bash
python -m ahu_paimon_toolkit.benchmark_vllm
```

## License