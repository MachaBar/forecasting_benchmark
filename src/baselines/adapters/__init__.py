from .chronos_adapter import ChronosAdapter
from .tsicl_adapter import TSICLAdapter
from .tirex_adapter import TiRexAdapter

ADAPTERS = {
    "chronos2": ChronosAdapter,
    "tsicl": TSICLAdapter,
    "tirex2": TiRexAdapter,
}

def get_adapter(cfg):
    key = cfg.model.name
    if key not in ADAPTERS:
        raise KeyError(f"unknown foundation model '{key}'. Available: {list(ADAPTERS)}")
    return ADAPTERS[key](cfg.model)