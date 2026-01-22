"""Load generation for CPU and GPU."""

import os
import subprocess
import time
import signal
from typing import Optional
from pathlib import Path
import importlib.resources


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
    """Control GPU load using gpu_load.py."""

    def stop(self) -> None:
        """
        Stop the GPU load process safely.
        """
        super().stop()

    def set_load(self, flags: str, duration: int = 3600) -> bool:
        """
        Set GPU load using specific flags.

        Args:
            flags: Command line flags for gpu_load.py (e.g. "--load 50")
            duration: How long to run in seconds

        Returns:
            True if started successfully
        """
        self.stop()

        flags = flags.strip()
        if not flags:
            return True

        try:
            # Get path to bundled gpu_load.py using importlib.resources
            # For Python 3.9+, use files() API
            try:
                from importlib.resources import files
                tools_path = files('data_collect.tools')
                script_path = tools_path / 'gpu_load.py'
            except ImportError:
                # Fallback for older Python versions (3.7-3.8)
                import importlib.resources as pkg_resources
                with pkg_resources.path('data_collect.tools', 'gpu_load.py') as p:
                    script_path = p

            # Split flags into list
            cmd_args = flags.split()
            cmd = ["uv", "run", str(script_path)] + cmd_args + ["-d", str(duration)]

            # Start in a new session to allow group termination
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True

        except FileNotFoundError:
            print("Error: uv not found. Install it first.")
            return False
        except Exception as e:
            print(f"Error starting GPU load: {e}")
            return False


class CPULoadController(LoadController):
    """Control CPU load using stress."""

    def __init__(self):
        super().__init__()
        self.total_cores = os.cpu_count() or 16

    def set_load(self, flags: str, duration: int = 3600) -> bool:
        """
        Set CPU load using specific flags.

        Args:
            flags: Command line flags for stress (e.g. "--cpu 6")
            duration: How long to run in seconds

        Returns:
            True if started successfully
        """
        self.stop()

        flags = flags.strip()
        if not flags:
            return True

        try:
            # Split flags into list
            cmd_args = flags.split()
            cmd = ["stress"] + cmd_args + ["--timeout", f"{duration}s"]

            # Start in a new session to allow group termination
            self.process = subprocess.Popen(
                cmd,
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
        self.current_cpu_load: Optional[str] = None
        self.current_gpu_load: Optional[str] = None

    def set_workload(self, cpu_flags: str, gpu_flags: str) -> bool:
        """
        Set combined CPU and GPU load.

        Returns:
            True if both loads started successfully
        """
        # Skip if load hasn't changed
        if self.current_cpu_load == cpu_flags and self.current_gpu_load == gpu_flags:
            return True

        cpu_ok = self.cpu_controller.set_load(cpu_flags)
        gpu_ok = self.gpu_controller.set_load(gpu_flags)

        if cpu_ok and gpu_ok:
            self.current_cpu_load = cpu_flags
            self.current_gpu_load = gpu_flags
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
            ["pkill", "-9", "-f", "gpu_load.py"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
