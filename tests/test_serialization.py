import pytest
from ritm_annotation.utils.serialization import get_class_from_str


def test_get_class_from_str_valid():
    # This should work as it is within the allowed namespace
    cls = get_class_from_str("ritm_annotation.model.is_model.ISModel")
    assert cls.__name__ == "ISModel"


def test_get_class_from_str_invalid():
    # These should fail with the security fix
    with pytest.raises(ValueError, match="Access to arbitrary modules is restricted"):
        get_class_from_str("os.system")

    with pytest.raises(ValueError, match="Access to arbitrary modules is restricted"):
        get_class_from_str("subprocess.Popen")
