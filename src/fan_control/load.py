"""Load generation for CPU and GPU."""

import os
import subprocess
import time
import signal
from typing import Optional


class LoadController:
    """Base class for load controllers."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None

    def stop(self) -> None:
        """Stop the load process and its children."""
        if self.process:
            try:
                # Send SIGTERM to the entire process group
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)

                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Fallback to SIGKILL if it doesn't stop
                    os.killpg(pgid, signal.SIGKILL)
                    self.process.wait()
            except ProcessLookupError:
                # Process already gone
                pass
            finally:
                self.process = None


class GPULoadController(LoadController):
    """Control GPU load using gpu-burn."""

    def stop(self) -> None:
        """
        Stop the GPU load process safely.

        We attempt to kill the child processes (workers) first while keeping
        the parent alive. This is because:
        1. gpu-burn parent exits immediately on SIGTERM without waiting for children
        2. If parent dies, the pipe to children closes
        3. Children writing to closed pipe get SIGPIPE and die immediately
        4. Immediate death skips cleanup (cuMemFree), leaving GPU in bad state

        By killing children first, they can cleanup and exit gracefully, causing
        the parent to exit naturally when it detects no active clients.
        """
        if self.process:
            try:
                # Attempt to kill children first using pkill
                # -P <ppid> matches processes whose parent is ppid
                subprocess.run(
                    ["pkill", "-TERM", "-P", str(self.process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

                # Give children time to cleanup and parent time to notice and exit
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # If graceful shutdown failed, will fall through to base stop()
                    pass
            except (FileNotFoundError, Exception):
                # If pkill missing or other error, fall through to base stop()
                pass

        super().stop()

    def set_load(self, percentage: int, duration: int = 3600) -> bool:
        """
        Set GPU load to specific percentage.

        Args:
            percentage: Target GPU load (0-100)
            duration: How long to run in seconds

        Returns:
            True if started successfully
        """
        self.stop()

        if percentage == 0:
            return True

        try:
            # gpu-burn runs at 100% by default
            # For percentage control, we use -m flag to limit memory usage
            # This indirectly controls GPU utilization
            # Start in a new session to allow group termination
            self.process = subprocess.Popen(
                ["gpu-burn", "-m", f"{percentage}%", str(duration)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True

        except FileNotFoundError:
            print("Error: gpu-burn not found. Install it first.")
            return False
        except Exception as e:
            print(f"Error starting GPU load: {e}")
            return False


class CPULoadController(LoadController):
    """Control CPU load using stress."""

    def __init__(self):
        super().__init__()
        self.total_cores = os.cpu_count() or 16

    def set_load(self, percentage: int, duration: int = 3600) -> bool:
        """
        Set CPU load to specific percentage.

        Args:
            percentage: Target CPU load (0-100)
            duration: How long to run in seconds

        Returns:
            True if started successfully
        """
        self.stop()

        if percentage == 0:
            return True

        try:
            # stress doesn't have --cpu-load like stress-ng
            # We control load by spawning a proportional number of workers
            cores_to_use = max(1, int(self.total_cores * percentage / 100))

            # Start in a new session to allow group termination
            self.process = subprocess.Popen(
                [
                    "stress",
                    "--cpu",
                    str(cores_to_use),
                    "--timeout",
                    f"{duration}s",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True

        except FileNotFoundError:
            print("Error: stress not found. Install it first.")
            return False
        except Exception as e:
            print(f"Error starting CPU load: {e}")
            return False


class LoadOrchestrator:
    """Coordinate CPU and GPU loads."""

    def __init__(self, load_stabilization_time: int = 10):
        self.load_stabilization_time = load_stabilization_time
        self.gpu_controller = GPULoadController()
        self.cpu_controller = CPULoadController()
        self.current_cpu_load: Optional[int] = None
        self.current_gpu_load: Optional[int] = None

    def set_workload(self, cpu_percent: int, gpu_percent: int) -> bool:
        """
        Set combined CPU and GPU load.

        Returns:
            True if both loads started successfully
        """
        # Skip if load hasn't changed
        if (
            self.current_cpu_load == cpu_percent
            and self.current_gpu_load == gpu_percent
        ):
            return True

        cpu_ok = self.cpu_controller.set_load(cpu_percent)
        gpu_ok = self.gpu_controller.set_load(gpu_percent)

        if cpu_ok and gpu_ok:
            self.current_cpu_load = cpu_percent
            self.current_gpu_load = gpu_percent
            # Wait for load to stabilize
            time.sleep(self.load_stabilization_time)
            return True

        self.current_cpu_load = None
        self.current_gpu_load = None
        return False

    def stop_all(self) -> None:
        """Stop all load generation."""
        self.cpu_controller.stop()
        self.gpu_controller.stop()
        self.current_cpu_load = None
        self.current_gpu_load = None

        # Also kill any stray processes
        subprocess.run(
            ["pkill", "-9", "stress"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["pkill", "-9", "gpu-burn"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
