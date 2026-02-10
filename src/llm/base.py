from abc import ABC, abstractmethod
# from .typing import ProviderResponse  # optional: define if you like
from ..app_types import ProviderRequest

class Provider(ABC):
    name: str
   
    def complete(self, req: ProviderRequest) -> str:
        r=self.client.chat.completions.create(
            model=req.model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            temperature=req.temperature,
        )
        return r.choices[0].message.content
