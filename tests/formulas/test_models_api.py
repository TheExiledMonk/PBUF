import inspect

from core import gr_models, pbuf_models


def test_model_api_signatures():
    functions = ["E2", "H", "Dc", "DL", "mu"]
    for name in functions:
        assert hasattr(gr_models, name)
        assert hasattr(pbuf_models, name)
        lcdm_sig = inspect.signature(getattr(gr_models, name))
        pbuf_sig = inspect.signature(getattr(pbuf_models, name))
        assert lcdm_sig.parameters.keys() == pbuf_sig.parameters.keys()
