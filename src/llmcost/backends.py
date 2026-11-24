from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import ollama

from .config import ModelCfg, RunCfg


@dataclass(slots=True)
class GenResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    stop_reason: str
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    gen_tokens_per_s: float | None = None
    prompt_tokens_per_s: float | None = None
    load_s: float | None = None
    error: str = ""


class Backend(Protocol):
    cfg: ModelCfg

    async def generate(self, prompt: str) -> GenResult: ...
    async def info(self) -> dict[str, Any]: ...
    async def aclose(self) -> None: ...


class AnthropicBackend:
    def __init__(self, cfg: ModelCfg, run: RunCfg) -> None:
        self.cfg, self.run = cfg, run
        self.client = anthropic.AsyncAnthropic(max_retries=6)

    async def generate(self, prompt: str) -> GenResult:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.run.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.cfg.thinking:
            kwargs["thinking"] = {"type": self.cfg.thinking}
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}
        if self.cfg.cache:
            kwargs["cache_control"] = {"type": "ephemeral"}

        t0 = time.perf_counter()
        try:
            msg = await self.client.messages.create(**kwargs)
        except (anthropic.BadRequestError, anthropic.AuthenticationError,
                anthropic.PermissionDeniedError, anthropic.NotFoundError):
            raise
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            return GenResult("", 0, 0, time.perf_counter() - t0, "error", error=f"{type(e).__name__}: {e}"[:300])
        latency = time.perf_counter() - t0

        u = msg.usage
        text = "".join(b.text for b in msg.content if b.type == "text")
        return GenResult(
            text=text,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            latency_s=latency,
            stop_reason=str(msg.stop_reason),
            cache_write_tokens=u.cache_creation_input_tokens or 0,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            gen_tokens_per_s=u.output_tokens / latency if latency > 0 else None,
        )

    async def info(self) -> dict[str, Any]:
        return {"sdk": f"anthropic {anthropic.__version__}"}

    async def aclose(self) -> None:
        await self.client.close()


class OllamaBackend:
    def __init__(self, cfg: ModelCfg, run: RunCfg) -> None:
        self.cfg, self.run = cfg, run
        self.client = ollama.AsyncClient(host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))

    async def generate(self, prompt: str) -> GenResult:
        options: dict[str, Any] = {"temperature": 0, "seed": self.run.seed, "num_predict": self.run.max_tokens}
        if self.cfg.num_ctx:
            options["num_ctx"] = self.cfg.num_ctx
        kwargs: dict[str, Any] = {}
        if self.cfg.think is not None:
            kwargs["think"] = self.cfg.think

        t0 = time.perf_counter()
        try:
            r = await self.client.chat(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive="60m",
                **kwargs,
            )
        except (ollama.ResponseError, ConnectionError) as e:
            return GenResult("", 0, 0, time.perf_counter() - t0, "error", error=f"{type(e).__name__}: {e}"[:300])
        latency = time.perf_counter() - t0

        def rate(count: int | None, ns: int | None) -> float | None:
            return count / (ns / 1e9) if count and ns else None

        return GenResult(
            text=r.message.content or "",
            input_tokens=r.prompt_eval_count or 0,
            output_tokens=r.eval_count or 0,
            latency_s=latency,
            stop_reason=str(r.done_reason),
            gen_tokens_per_s=rate(r.eval_count, r.eval_duration),
            prompt_tokens_per_s=rate(r.prompt_eval_count, r.prompt_eval_duration),
            load_s=(r.load_duration or 0) / 1e9,
        )

    async def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sdk": "ollama-python"}
        listed = (await self.client.list()).model_dump()
        for m in listed.get("models", []):
            if m.get("model") == self.cfg.model:
                out["weights_bytes"] = m.get("size")
                out["digest"] = m.get("digest")
                out["details"] = m.get("details")
        running = (await self.client.ps()).model_dump()
        for m in running.get("models", []):
            if m.get("model") == self.cfg.model:
                out["resident_bytes"] = m.get("size")
                out["resident_vram_bytes"] = m.get("size_vram")
        return out

    async def aclose(self) -> None:
        pass


def make_backend(cfg: ModelCfg, run: RunCfg) -> Backend:
    match cfg.backend:
        case "anthropic":
            return AnthropicBackend(cfg, run)
        case "ollama":
            return OllamaBackend(cfg, run)
    raise ValueError(cfg.backend)
