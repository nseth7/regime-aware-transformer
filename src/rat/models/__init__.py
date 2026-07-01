from rat.models.baselines import BaselineTransformer, MacroConcatTransformer
from rat.models.rat import RegimeAwareTransformer

MODEL_REGISTRY = {
    "regime": RegimeAwareTransformer,
    "baseline": BaselineTransformer,
    "macro_concat": MacroConcatTransformer,
}

__all__ = [
    "RegimeAwareTransformer",
    "BaselineTransformer",
    "MacroConcatTransformer",
    "MODEL_REGISTRY",
]
