"""
config.py

统一大模型调用接口层配置。
只管理配置，不写真实 API Key。

兼容早期岗位流水线使用的环境变量（与当时独立脚本约定一致）：
- JOB_AGENT_LLM_MODE=mock | openai_compatible（若设置则优先于 LLM_MOCK_ENABLED）
- LLM_BASE_URL（在 LLM_API_BASE_URL 未设置时作为回退）
- LLM_MODEL（在 LLM_MODEL_NAME 未设置时作为回退）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_mock_enabled() -> bool:
    legacy = os.getenv("JOB_AGENT_LLM_MODE", "").strip().lower()
    if legacy == "mock":
        return True
    if legacy == "openai_compatible":
        return False
    return _truthy_env("LLM_MOCK_ENABLED", "1")


def _resolve_api_base_url() -> str:
    return os.getenv("LLM_API_BASE_URL", "").strip() or os.getenv("LLM_BASE_URL", "").strip()


def _resolve_model_name() -> str:
    return (
        os.getenv("LLM_MODEL_NAME", "").strip()
        or os.getenv("LLM_MODEL", "").strip()
        or "mock-gpt"
    )


_MOCK_ENABLED = _resolve_mock_enabled()
_API_BASE_URL = _resolve_api_base_url()
_MODEL_NAME = _resolve_model_name()


@dataclass(frozen=True)
class LLMConfig:
    """大模型客户端配置。"""

    model_name: str = _MODEL_NAME
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    retry_times: int = int(os.getenv("LLM_RETRY_TIMES", "2"))
    mock_enabled: bool = _MOCK_ENABLED
    api_base_url: str = _API_BASE_URL
    api_key_env_name: str = os.getenv("LLM_API_KEY_ENV_NAME", "LLM_API_KEY")


@dataclass(frozen=True)
class StateConfig:
    """学生主状态文件配置。"""

    default_state_path: Path = Path(
        os.getenv("STUDENT_STATE_PATH", "outputs/state/student.json")
    )
    encoding: str = "utf-8"
    indent: int = 2


DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_STATE_CONFIG = StateConfig()
