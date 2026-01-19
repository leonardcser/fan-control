#!/usr/bin/env python3
"""
GPU Load Generator with controllable utilization and memory usage.

Usage:
    python gpu_load.py --load 50 --memory 30 --duration 60
    python gpu_load.py --load 80 --memory 50  # Run indefinitely
    python gpu_load.py --sweep  # Sweep through load levels for testing

This gives finer control than gpu-burn for data collection.
"""

import argparse
import time
import signal
import sys

import torch

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False


def get_gpu_info():
    """Get GPU memory info."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)

    return {
        "total_gb": total / 1e9,
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "free_gb": (total - reserved) / 1e9,
    }


def allocate_memory(target_percent: float, reserved_gb: float = 1.0) -> list[torch.Tensor]:
    """
    Allocate GPU memory to reach target percentage of total VRAM.

    Args:
        target_percent: Target percentage of TOTAL VRAM to use (0-100)
        reserved_gb: GB to reserve for compute workload and PyTorch overhead

    Returns:
        List of tensors (kept as list to avoid reallocation during concat)
    """
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_bytes / 1e9

    # Calculate how much to allocate
    # target_percent of total, minus what we reserve for compute
    target_gb = (total_gb * target_percent / 100) - reserved_gb
    target_gb = max(0, target_gb)

    if target_gb <= 0:
        print(f"  Target memory ({target_percent}%) leaves no room after {reserved_gb}GB reserved")
        return []

    target_bytes = int(target_gb * 1e9)

    # Allocate in chunks to avoid OOM and fragmentation
    chunk_size_bytes = 512 * 1024 * 1024  # 512MB chunks
    tensors = []
    allocated_bytes = 0

    while allocated_bytes < target_bytes:
        remaining = target_bytes - allocated_bytes
        chunk_bytes = min(chunk_size_bytes, remaining)
        num_floats = chunk_bytes // 4

        try:
            t = torch.zeros(num_floats, dtype=torch.float32, device="cuda")
            tensors.append(t)
            allocated_bytes += chunk_bytes
        except torch.cuda.OutOfMemoryError:
            print(f"  Memory allocation stopped at {allocated_bytes / 1e9:.2f} GB (OOM)")
            break

    return tensors


def create_workload(size: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Create matrices for compute workload."""
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    return a, b


def run_compute_burst(a: torch.Tensor, b: torch.Tensor, iterations: int = 10):
    """Run matrix multiplications."""
    c = None
    for _ in range(iterations):
        c = torch.mm(a, b)
        torch.cuda.synchronize()
    return c


def run_load(
    load_percent: float,
    memory_percent: float,
    duration: float | None = None,
    matrix_size: int = 4096,
    report_interval: float = 5.0,
):
    """
    Run GPU load at specified utilization and memory levels.

    Args:
        load_percent: Target GPU compute utilization (0-100)
        memory_percent: Target VRAM usage (0-100)
        duration: Run time in seconds (None = indefinite)
        matrix_size: Size of matrices for compute (larger = more load per iteration)
        report_interval: How often to print status
    """
    global running

    print(f"GPU Load Generator")
    print(f"  Target Load: {load_percent}%")
    print(f"  Target Memory: {memory_percent}%")
    print(f"  Duration: {duration}s" if duration else "  Duration: indefinite")
    print()

    # Create compute workload FIRST (to know how much memory it needs)
    print("Creating compute workload...")
    a, b = create_workload(matrix_size)
    torch.cuda.synchronize()
    workload_mem = torch.cuda.memory_allocated(0) / 1e9
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Workload memory: {workload_mem:.2f} GB")
    print()

    # Now allocate additional memory to reach target
    print("Allocating memory...")
    # Reserve extra for PyTorch overhead + workspace
    reserved = workload_mem + 0.5
    mem_tensors = allocate_memory(memory_percent, reserved_gb=reserved)
    info = get_gpu_info()
    print(f"  Total allocated: {info['allocated_gb']:.1f} GB / {info['total_gb']:.1f} GB ({info['allocated_gb']/info['total_gb']*100:.0f}%)")
    print()

    # Calibrate timing
    # Run a burst and measure time to calibrate duty cycle
    print("Calibrating...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_compute_burst(a, b, iterations=5)
    burst_time = time.perf_counter() - t0
    print(f"  Burst time (5 iters): {burst_time*1000:.1f}ms")

    # Calculate duty cycle
    # load_percent = burst_time / (burst_time + sleep_time) * 100
    # sleep_time = burst_time * (100 / load_percent - 1)
    if load_percent >= 99:
        sleep_time = 0
        iterations_per_burst = 50  # Continuous compute
    elif load_percent <= 1:
        sleep_time = 1.0
        iterations_per_burst = 1
    else:
        iterations_per_burst = 5
        burst_time_est = burst_time  # 5 iterations
        sleep_time = burst_time_est * (100 / load_percent - 1)
        sleep_time = max(0.001, min(sleep_time, 1.0))  # Clamp

    print(f"  Sleep time: {sleep_time*1000:.1f}ms")
    print(f"  Iterations per burst: {iterations_per_burst}")
    print()

    print("Running... (Ctrl+C to stop)")
    print("-" * 50)

    start_time = time.time()
    last_report = start_time
    total_compute_time = 0
    total_sleep_time = 0

    try:
        while running:
            # Check duration
            elapsed = time.time() - start_time
            if duration and elapsed >= duration:
                break

            # Compute burst
            t0 = time.perf_counter()
            run_compute_burst(a, b, iterations=iterations_per_burst)
            compute_time = time.perf_counter() - t0
            total_compute_time += compute_time

            # Sleep
            if sleep_time > 0:
                time.sleep(sleep_time)
                total_sleep_time += sleep_time

            # Report
            if time.time() - last_report >= report_interval:
                actual_load = total_compute_time / (total_compute_time + total_sleep_time) * 100
                info = get_gpu_info()
                print(
                    f"  Elapsed: {elapsed:.0f}s | "
                    f"Actual Load: {actual_load:.1f}% | "
                    f"Memory: {info['allocated_gb']:.1f} GB"
                )
                last_report = time.time()

    except KeyboardInterrupt:
        pass

    # Cleanup
    print()
    print("Cleaning up...")
    del a, b
    for t in mem_tensors:
        del t
    mem_tensors.clear()
    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    if total_compute_time + total_sleep_time > 0:
        actual_load = total_compute_time / (total_compute_time + total_sleep_time) * 100
    else:
        actual_load = 0
    print(f"Done. Ran for {elapsed:.1f}s at ~{actual_load:.1f}% load")


def sweep_loads(memory_percent: float = 30, step_duration: float = 30):
    """Sweep through different load levels for testing."""
    global running

    load_levels = [20, 40, 60, 80, 100]

    print("Load Sweep Mode")
    print(f"  Levels: {load_levels}")
    print(f"  Step duration: {step_duration}s each")
    print(f"  Memory: {memory_percent}%")
    print()

    for load in load_levels:
        if not running:
            break
        print(f"\n{'='*50}")
        print(f"LOAD LEVEL: {load}%")
        print(f"{'='*50}")
        run_load(load, memory_percent, duration=step_duration, report_interval=10)

    print("\nSweep complete!")


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="GPU Load Generator with controllable utilization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --load 50 --memory 30          # 50%% GPU, 30%% VRAM, run forever
  %(prog)s --load 80 --memory 50 -d 60    # 80%% load for 60 seconds
  %(prog)s --sweep                         # Sweep through load levels
  %(prog)s --load 100 --memory 80         # Max stress test
        """,
    )

    parser.add_argument(
        "--load", "-l",
        type=float,
        default=50,
        help="Target GPU compute utilization (0-100, default: 50)",
    )
    parser.add_argument(
        "--memory", "-m",
        type=float,
        default=30,
        help="Target VRAM usage percentage (0-100, default: 30)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Duration in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--matrix-size", "-s",
        type=int,
        default=4096,
        help="Matrix size for compute workload (default: 4096)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep through load levels (20, 40, 60, 80, 100%%)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    info = get_gpu_info()
    print(f"VRAM: {info['total_gb']:.1f} GB total, {info['free_gb']:.1f} GB free")
    print()

    if args.sweep:
        sweep_loads(memory_percent=args.memory, step_duration=args.duration or 30)
    else:
        run_load(
            load_percent=args.load,
            memory_percent=args.memory,
            duration=args.duration,
            matrix_size=args.matrix_size,
        )


if __name__ == "__main__":
    main()
