# src/baselines/adapters/__init__.py
from .chronos_adapter import ChronosAdapter
from .tsicl_adapter import TSICLAdapter
from .tirex_adapter import TiRexAdapter
from .patchtst_fm_adapter import PatchTSTFMAdapter
from .tabicl_adapter import TabICLAdapter

ADAPTERS = {
    "chronos2": ChronosAdapter,
    "tsicl": TSICLAdapter,
    "tirex2": TiRexAdapter,
    "patchtst_fm": PatchTSTFMAdapter,
    "tabicl": TabICLAdapter,
}

def get_adapter(cfg):
    key = cfg.model.name
    if key not in ADAPTERS:
        raise KeyError(f"unknown foundation model '{key}'. Available: {list(ADAPTERS)}")
    return ADAPTERS[key](cfg.model)