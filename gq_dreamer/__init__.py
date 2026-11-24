"""GQ-Dreamer for Super Mario Bros. Deluxe: a GroundQuery tokenizer + transformer world model (TSSM) with an
actor-critic trained in imagination."""
import importlib

# Lazy exports (PEP 562): env worker processes only import gq_dreamer.env, so they do not load torch or matplotlib.
_EXPORTS = {"run": ".trainer", "Trainer": ".trainer", "GQDreamer": ".gq_dreamer", "create_envs": ".env",
            "GQDreamerConfig": ".config", "model_config": ".config", "config_from_dict": ".config",
            "GQ_CONFIG": ".config", "MODEL_CONFIGS": ".config"}
__all__ = list(_EXPORTS)


def __getattr__(name):
    """Imports an exported name from its module on first use."""
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
