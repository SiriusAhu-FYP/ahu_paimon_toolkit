
import pynvml
from importlib import resources
from loguru import logger as lg
import base64
import subprocess
import time

def get_resource_path(resource_name):
    """Get the path to the resource."""
    # Locate the resource in the assets folder
    traversable = resources.files("ahu_paimon_toolkit.assets").joinpath(resource_name)

    # Get absolute path
    with resources.as_file(traversable) as img_path:
        final_path = str(img_path)
        lg.debug(f"final_path: {final_path}")
        return final_path

json_output_template = """
{
    bounding_box: [
    {
        "bbox_2d": [111, 222, 333, 444], 
        "label": "dog",
    },
    {
        "bbox_2d": [555, 666, 777, 888], 
        "label": "cat",
    },
    ...
    ],
    response: "I found a dog and a cat in the picture. The dog seems to be a ..."
}
"""

def get_gpu_memory():
    """Get the current GPU memory usage (MB) using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0

def encode_image(image_path):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def benchmark_report_generator(metrics: dict, model_name: str, device_info: str):
    """
    Generate the full report of the benchmark (average values). 
    Including:
    - Basic Information
        - Time (Date, Time)
        - Model Name
        - Device parameters (GPU name & VRAM)
    - Metrics (Benchmark results)
        - Throughput
        - Total Time
        - VRAM Used
        - Output Tokens
        - Input Tokens
        - Total Tokens
    """
    report = f"==== Benchmark Report ({model_name}) ====\n"
    report += f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"Model Name: {model_name}\n"
    report += f"Device: {device_info}\n"
    report += f"Metrics (Average):\n"
    report += f" - Throughput: {sum(metrics['throughput']) / len(metrics['throughput']):.2f}\n"
    report += f" - Total Time: {sum(metrics['total_time']) / len(metrics['total_time']):.2f}\n"
    report += f" - VRAM Used: {sum(metrics['vram_used']) / len(metrics['vram_used']):.2f}\n"
    report += f" - Output Tokens: {sum(metrics['output_tokens']) / len(metrics['output_tokens']):.2f}\n"
    return report

def get_device_info():
    """Get the device information."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    result = f"GPU: {name} | Total VRAM: {info.total / 1024**2:.2f} MB | Used VRAM: {info.used / 1024**2:.2f} MB | Free VRAM: {info.free / 1024**2:.2f} MB"
    pynvml.nvmlShutdown()
    return result