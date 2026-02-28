from loguru import logger as lg
from .utils import json_output_template, get_gpu_memory, encode_image, benchmark_report_generator, get_resource_path, get_device_info
from openai import OpenAI
import time

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_IMAGE_PATH = "cat_park.jpg"
DEFAULT_PROMPT = f"请用中文详细描述一下这张图片里的内容。并给出bounding box。Json格式为：{json_output_template}"
DEFAULT_SAVE_PATH = None
DEFAULT_DO_STREAM = True
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512


def vllm_benchmark(num_runs: int = 3, warmup_runs: int = 1, model_name: str = DEFAULT_MODEL_NAME, api_base: str = DEFAULT_API_BASE, image_path: str = DEFAULT_IMAGE_PATH, prompt: str = DEFAULT_PROMPT, save_path: str = DEFAULT_SAVE_PATH, doStream: bool = DEFAULT_DO_STREAM, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS):
    """
    Benchmark the performance of a vLLM server.
    """
    # Resolve the image path to the absolute path and encode the image to base64
    img_path = get_resource_path(image_path)
    base64_image = encode_image(img_path)

    # Get the device
    device_info = get_device_info()

    # Initialize the client
    client = OpenAI(base_url=api_base, api_key="EMPTY")

    # Start the benchmark
    lg.info(f"Starting benchmark for {model_name} | Number of runs: {num_runs}")

    # Initialize the metrics dictionary
    metrics = {"ttft": [], "throughput": [], "total_time": [], "vram_used": [], "output_tokens": []}

    # Warmup
    if warmup_runs > 0:
        for _ in range(warmup_runs):
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ]},
                ],
            )
            lg.info(f"Warmup run {_ + 1} completed")

    # Benchmark
    for i in range(num_runs):
        start_time = time.perf_counter()
        vram_before = get_gpu_memory()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ]},
            ],
            stream=doStream,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_options={
                "include_usage": True
            },
        )

        ttft = None
        output_tokens = 0

        # Receive stream data
        for chunk in response:
            if ttft is None and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - start_time
                if num_runs == 1:
                    lg.info(f"💬 Model output: ", end="")
            if num_runs == 1 and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                lg.info(chunk.choices[0].delta.content, end="", flush=True)
            if chunk.usage is not None:
                output_tokens = chunk.usage.completion_tokens

        if num_runs == 1:
            lg.info("\n")

        lg.info(f"==== Run {i + 1} completed ====")

        # Calculate the metrics for the current run
        total_time = time.perf_counter() - start_time
        generation_time = total_time - ttft
        throughput = output_tokens / generation_time if generation_time > 0 else 0
        vram_after = get_gpu_memory()

        # Log the current run
        lg.info(f"Time to First Token (TTFT) (s): {ttft:.3f}")
        lg.info(f"Generation Throughput (tokens/s): {throughput:.1f}")
        lg.info(f"Output Tokens: {output_tokens}")
        lg.info(f"VRAM Used: {vram_after} MB (Request before: {vram_before} MB)")

        # Record the metrics
        metrics["ttft"].append(ttft)
        metrics["throughput"].append(throughput)
        metrics["total_time"].append(total_time)
        metrics["vram_used"].append(vram_after)
        metrics["output_tokens"].append(output_tokens)

    # Generate report (per run & average)
    report = benchmark_report_generator(metrics, model_name, device_info)
    print(report)

    # Save report (and raw data csv) to file
    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        lg.info(f"Report saved to {save_path}")

    return report