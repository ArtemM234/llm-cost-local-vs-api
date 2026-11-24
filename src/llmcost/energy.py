from __future__ import annotations

import asyncio
import platform
import plistlib
import shutil
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PowerSample:
    t_end: float
    duration_s: float
    watts: float


def parse_sample(d: dict, fallback_interval_s: float) -> tuple[float, float] | None:
    duration = d.get("elapsed_ns", 0) / 1e9 or fallback_interval_s
    proc = d.get("processor", {})
    if "combined_power" in proc:
        return duration, proc["combined_power"] / 1000
    parts = [proc.get(k) for k in ("cpu_power", "gpu_power", "ane_power")]
    if any(p is not None for p in parts):
        gpu = parts[1] if parts[1] is not None else d.get("gpu", {}).get("gpu_power", 0)
        return duration, ((parts[0] or 0) + gpu + (parts[2] or 0)) / 1000
    energies = [proc.get(k) for k in ("cpu_energy", "gpu_energy", "ane_energy")]
    if any(e is not None for e in energies):
        return duration, sum(e or 0 for e in energies) / 1000 / duration
    return None


def parse_nvidia_line(line: str) -> float | None:
    try:
        return float(line.strip())
    except ValueError:
        return None


def integrate(samples: list[PowerSample], a: float, b: float) -> tuple[float, float]:
    joules = covered = 0.0
    for s in samples:
        lo, hi = max(a, s.t_end - s.duration_s), min(b, s.t_end)
        if hi > lo:
            joules += s.watts * (hi - lo)
            covered += hi - lo
    return joules, covered


class PowerSampler:
    source = "base"
    tool = ""

    def __init__(self, interval_ms: int = 200) -> None:
        self.interval_ms = interval_ms
        self.samples: list[PowerSample] = []
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self.parse_failures = 0

    def _command(self) -> list[str]:
        raise NotImplementedError

    async def _read(self) -> None:
        raise NotImplementedError

    async def start(self) -> None:
        if shutil.which(self.tool) is None:
            raise RuntimeError(f"{self.tool} не найден (запустите с --no-energy)")
        self._proc = await asyncio.create_subprocess_exec(
            *self._command(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        self._reader = asyncio.create_task(self._read())
        try:
            await self.wait_until(time.monotonic(), timeout_s=5)
        except TimeoutError:
            await self.stop()
            raise RuntimeError(f"{self.tool} не выдаёт данных. {self._hint()}") from None

    def _hint(self) -> str:
        return ""

    def _append(self, watts: float, duration_s: float) -> None:
        self.samples.append(PowerSample(time.monotonic(), duration_s, watts))

    async def wait_until(self, t: float, timeout_s: float | None = None) -> None:
        deadline = time.monotonic() + (timeout_s if timeout_s is not None else 3 * self.interval_ms / 1000 + 2)
        while not self.samples or self.samples[-1].t_end < t:
            if time.monotonic() > deadline:
                raise TimeoutError
            await asyncio.sleep(self.interval_ms / 4000)

    async def energy_j(self, a: float, b: float) -> tuple[float, float]:
        await self.wait_until(b)
        return integrate(self.samples, a, b)

    async def idle_watts(self, seconds: float) -> float:
        a = time.monotonic()
        await asyncio.sleep(seconds)
        joules, covered = await self.energy_j(a, time.monotonic())
        return joules / covered if covered else 0.0

    async def stop(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), 5)
            except TimeoutError:
                self._proc.kill()
        if self._reader:
            self._reader.cancel()


class PowerMetricsSampler(PowerSampler):
    source = "powermetrics: SoC (CPU+GPU+ANE)"
    tool = "powermetrics"

    def _command(self) -> list[str]:
        return ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
                "-i", str(self.interval_ms), "-f", "plist"]

    def _hint(self) -> str:
        return "Сделайте `sudo -v` и повторите."

    async def _read(self) -> None:
        assert self._proc and self._proc.stdout
        buf = b""
        while chunk := await self._proc.stdout.read(65536):
            buf += chunk
            *docs, buf = buf.split(b"\x00")
            for doc in docs:
                if not (doc := doc.strip()):
                    continue
                try:
                    parsed = parse_sample(plistlib.loads(doc), self.interval_ms / 1000)
                except Exception:
                    parsed = None
                if parsed is None:
                    self.parse_failures += 1
                    continue
                self._append(parsed[1], parsed[0])


class NvidiaSmiSampler(PowerSampler):
    source = "nvidia-smi: GPU board power"
    tool = "nvidia-smi"

    def __init__(self, interval_ms: int = 100, gpu_index: int = 0) -> None:
        super().__init__(interval_ms)
        self.gpu_index = gpu_index

    def _command(self) -> list[str]:
        return ["nvidia-smi", "-i", str(self.gpu_index), "--query-gpu=power.draw.instant",
                "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)]

    async def _read(self) -> None:
        assert self._proc and self._proc.stdout
        last: float | None = None
        while line := await self._proc.stdout.readline():
            watts = parse_nvidia_line(line.decode(errors="ignore"))
            now = time.monotonic()
            if watts is None:
                self.parse_failures += 1
                continue
            gap = now - last if last is not None else self.interval_ms / 1000
            last = now
            self._append(watts, min(gap, 5 * self.interval_ms / 1000))


def make_sampler() -> PowerSampler:
    if platform.system() == "Darwin":
        return PowerMetricsSampler()
    if shutil.which("nvidia-smi"):
        return NvidiaSmiSampler()
    raise RuntimeError("нет датчика мощности: нужен macOS (powermetrics) или NVIDIA (nvidia-smi); запустите с --no-energy")
