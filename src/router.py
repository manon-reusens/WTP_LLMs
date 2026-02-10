import yaml
from pathlib import Path
from .llm.openai_provider import OpenAIProvider
from .llm.google_provider import GoogleProvider
from .llm.anthropic_provider import AnthropicProvider
from .llm.hf_provider import HFProvider
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "app.yaml").read_text())
PROV = yaml.safe_load((ROOT / "config" / "providers.yaml").read_text())
load_dotenv(ROOT / ".env")

def get_provider(name: str):
    name = name.lower()
    if name == "openai": return OpenAIProvider()
    if name == "google": return GoogleProvider()
    if name == "anthropic": return AnthropicProvider()
    if name == "huggingface": return HFProvider()
    raise ValueError(f"Unknown provider: {name}")

def resolve_model(provider_name: str, alias_or_id: str) -> str:
    # allows using friendly aliases from providers.yaml OR raw ids
    p = PROV.get(provider_name, {}).get("models", {})
    return p.get(alias_or_id, alias_or_id)

def default_config():
    return CONFIG["default_provider"], CONFIG["default_model"], CONFIG.get("temperature", 0)
