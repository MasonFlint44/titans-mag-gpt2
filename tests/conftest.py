import pytest
import torch


# G280: match the training-time TF32 setting in `train.py::main`. Fp32
# matmuls use 10-bit mantissa inputs on tensor cores; accumulation stays
# fp32. Affects NS5 and the analytical-grad LayerNorm internals — the
# only places that explicitly force fp32 (everything else runs under
# bf16 autocast and is unaffected). Setting this in conftest catches
# any TF32-precision-sensitive test failure before it hits production
# training. Process-global; applies to every test in this suite.
torch.set_float32_matmul_precision("high")


@pytest.fixture(autouse=True)
def deterministic_rng():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield
