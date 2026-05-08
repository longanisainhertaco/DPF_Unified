import inspect

import pytest

from dpf.validation import lee_model_comparison


def test_lee_model_reflected_shock_uses_kr_rankine_hugoniot_ratio() -> None:
    assert lee_model_comparison._D2_STRONG_SHOCK_COMPRESSION == pytest.approx(4.0)

    source = inspect.getsource(lee_model_comparison.LeeModel.run)
    assert "8.0 * rho0" not in source
    assert "_D2_STRONG_SHOCK_COMPRESSION * rho0" in source
