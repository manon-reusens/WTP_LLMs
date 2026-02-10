from dataclasses import dataclass
from typing import List, Literal, Dict, Any

Role = Literal["system", "user", "assistant"]

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class AppConfig:
    default_provider: str
    default_model: str
    temperature: float = 0

@dataclass
class ProviderRequest:
    model: str
    messages: List[Message]
    temperature: float
    extra: Dict[str, Any] | None = None
