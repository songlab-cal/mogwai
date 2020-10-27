from typing import Type, Dict
from .gremlin import Gremlin
from .base_model import BaseModel


MODELS: Dict[str, BaseModel] = {
    "gremlin": Gremlin,
}


def get(name: str) -> Type[BaseModel]:
    return MODELS[name.lower()]
