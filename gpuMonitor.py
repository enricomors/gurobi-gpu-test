import time
import threading
from gpustat import GPUStatCollection

"""Utilities to monitor GPU usage."""

stop_monitoring = False
gpu_stats = []
monitor_thread = None


def monitor_gpu_usage(interval=0.01):
    """
    Monitors the GPU usage on the current device.

    :param interval: interval in seconds of GPU monitor updates.
    """
    global gpu_stats, stop_monitoring
    gpu_stats = []
    while not stop_monitoring:
        try:
            stats = GPUStatCollection.new_query()
            # Assuming only one GPU, if there are multiple, you can adjust as needed
            gpu_memory = stats.gpus[0].memory_used if stats.gpus else 0
            gpu_stats.append(gpu_memory)
        except Exception as e:
            print(f"Error monitoring GPU: {e}")
        time.sleep(interval)


def start_monitoring(interval=0.01):
    """
    Starts the GPU monitoring in a separate thread.

    :param interval: interval in seconds of GPU monitor updates.
    """
    global monitor_thread, stop_monitoring
    stop_monitoring = False
    monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(interval,))
    monitor_thread.start()


def stop_monitoring_gpu():
    """
    Stops the GPU monitoring.
    """
    global stop_monitoring, monitor_thread
    stop_monitoring = True
    if monitor_thread:
        monitor_thread.join()


def get_latest_gpu_usage():
    """
    Returns the most recent GPU usage sample.

    :return: Latest GPU memory usage in MB, or None if no data available.
    """
    if gpu_stats:
        return gpu_stats[-1]
    return None


def is_gpu_available():
    """
    Checks if there's a GPU available on the current device.

    :return: True if GPU is available, False otherwise.
    """
    try:
        gpu = GPUStatCollection.new_query().gpus[0]
        # print(f"available gpu: {gpu}")
        return True
    except:
        return False
