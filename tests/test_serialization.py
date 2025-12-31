import pytest
from ritm_annotation.utils.serialization import get_class_from_str


def test_get_class_from_str_safelist():
    with pytest.raises(ValueError):
        get_class_from_str("os.system")


def test_get_class_from_str_allowed():
    get_class_from_str("ritm_annotation.model.is_hrnet_model.HRNetModel")
