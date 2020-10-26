from typing import Type, Dict
from .gremlin import Gremlin
from .attention import Attention
from .factored_attention import FactoredAttention
from .base_model import BaseModel


MODELS: Dict[str, BaseModel] = {
    "gremlin": Gremlin,
    "attention": Attention,
    "factored_attention": FactoredAttention,
}


def get(name: str) -> Type[BaseModel]:
    return MODELS[name.lower()]
