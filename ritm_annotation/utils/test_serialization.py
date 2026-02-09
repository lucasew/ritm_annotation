import pytest
from ritm_annotation.utils.serialization import get_class_from_str
import torch


def test_get_class_from_str_allowed():
    """Test that allowed classes can be loaded."""
    # Test ritm_annotation prefix
    cls = get_class_from_str("ritm_annotation.utils.serialization.get_class_from_str")
    assert cls is get_class_from_str

    # Test torch prefix
    cls = get_class_from_str("torch.nn.Linear")
    assert cls is torch.nn.Linear

    # Test isegm prefix (if it works or fails cleanly on import, but security check should pass)
    try:
        get_class_from_str("isegm.model.is_model.ISModel")
    except ModuleNotFoundError:
        # Expected since isegm is not installed, but security check passed
        pass
    except ValueError as e:
        pytest.fail(f"Should not raise ValueError for allowed prefix: {e}")


def test_get_class_from_str_disallowed():
    """Test that disallowed classes raise ValueError."""
    # Test os module
    try:
        get_class_from_str("os.system")
        pytest.fail("Should have raised ValueError for os.system")
    except ValueError as e:
        assert "Security violation" in str(e)
    except Exception as e:
        pytest.fail(
            f"Should have raised ValueError, but raised {type(e).__name__}: {e}"
        )

    # Test subprocess
    try:
        get_class_from_str("subprocess.call")
        pytest.fail("Should have raised ValueError for subprocess.call")
    except ValueError as e:
        assert "Security violation" in str(e)
    except Exception as e:
        pytest.fail(
            f"Should have raised ValueError, but raised {type(e).__name__}: {e}"
        )

    # Test pickle
    try:
        get_class_from_str("pickle.load")
        pytest.fail("Should have raised ValueError for pickle.load")
    except ValueError as e:
        assert "Security violation" in str(e)
    except Exception as e:
        pytest.fail(
            f"Should have raised ValueError, but raised {type(e).__name__}: {e}"
        )
