# Tests for RITM Annotation

Comprehensive test suite for RITM Annotation modular architecture.

## Test Organization

```
tests/
├── conftest.py                              # Shared fixtures and utilities
├── core/                                    # Core component tests
│   ├── test_annotation_utils.py            # Pure function tests
│   ├── test_annotation_session.py          # AnnotationSession unit tests
│   ├── test_annotation_session_integration.py  # Integration tests with real model
│   ├── test_batch_processor.py             # BatchProcessor tests
│   └── test_training_loop.py               # TrainingLoop tests
└── test_end_to_end.py                      # End-to-end workflow tests
```

## Test Categories

### Unit Tests (Fast)
Tests for pure functions and components with mocked dependencies.

- `test_annotation_utils.py`: Pure utility functions
- `test_annotation_session.py`: AnnotationSession with mock predictor

Run with:
```bash
pytest -m unit
```

### Integration Tests (Slower)
Tests using real PyTorch models and components.

- `test_annotation_session_integration.py`: Real model predictions
- `test_batch_processor.py`: Real batch processing
- `test_training_loop.py`: Real training loops
- `test_end_to_end.py`: Complete workflows

Run with:
```bash
pytest -m integration
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest ritm_annotation/tests/core/test_annotation_utils.py
```

### Run specific test
```bash
pytest ritm_annotation/tests/core/test_annotation_utils.py::TestPureFunctions::test_compute_iou
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage
```bash
pytest --cov=ritm_annotation --cov-report=html
```

### Run only fast tests (unit)
```bash
pytest -m "not integration"
```

### Run only integration tests
```bash
pytest -m integration
```

## Test Fixtures

### Available Fixtures

- **device**: CPU device for testing (consistent results)
- **test_model**: Small HRNet18 model for testing (no pretrained weights)
- **test_image**: Small 100x100 test image
- **test_image_large**: Larger 480x640 test image
- **test_mask**: Synthetic mask with two objects
- **test_batch**: Batch of synthetic data
- **mock_predictor**: Mock predictor with realistic behavior
- **real_predictor**: Real predictor using test_model
- **temp_checkpoint_dir**: Temporary directory for checkpoints
- **simple_loss_fn**: Loss functions for testing
- **simple_metric**: IoU metric for testing

### Using Fixtures

```python
def test_example(test_model, test_image, device):
    model, model_cfg = test_model
    # Use model and image in test
```

## Utilities

### Pure Functions Test Utilities

```python
from ritm_annotation.tests.conftest import (
    assert_tensor_equal,
    assert_valid_probability_map,
    create_synthetic_dataset,
)

# Check tensors are equal
assert_tensor_equal(tensor1, tensor2)

# Validate probability map
assert_valid_probability_map(prob_map)

# Create synthetic dataset
dataset = create_synthetic_dataset(num_samples=10, image_size=(320, 480))
```

## What is Tested

### AnnotationSession
- Image loading
- Click addition and removal
- Undo/redo functionality
- Object finishing
- Mask management
- State consistency
- Event emission
- Visualization data

### TrainingLoop
- Single and multiple epoch training
- Training vs evaluation mode
- Loss computation and decrease
- Checkpoint saving/loading
- Callbacks
- Early stopping
- Gradient updates
- Learning rate scheduling

### BatchProcessor
- Batch processing
- Loss computation
- Metric updates
- Click simulation
- Gradient computation
- Numerical stability

### Pure Utility Functions
- Mask thresholding
- Mask merging
- IoU computation
- Image/mask validation
- Click statistics
- Object center estimation
- Boundary point detection
- Image blending
- Click visualization

### End-to-End Workflows
- Complete annotation workflow
- Save/load masks
- Interactive refinement
- Training and resume
- Model persistence
- GUI adapter integration
- Data integrity

## Test Guidelines

### Writing New Tests

1. **Use appropriate fixtures**
   ```python
   def test_something(test_model, device):
       # Test implementation
   ```

2. **Mark tests appropriately**
   ```python
   @pytest.mark.integration
   def test_with_real_model():
       pass

   @pytest.mark.slow
   def test_long_running():
       pass
   ```

3. **Test edge cases**
   ```python
   def test_empty_input():
       # Test with empty data

   def test_single_pixel():
       # Test with minimal data

   def test_large_input():
       # Test with large data
   ```

4. **Test error conditions**
   ```python
   def test_invalid_input():
       with pytest.raises(ValueError):
           function_with_invalid_input()
   ```

5. **Ensure reproducibility**
   ```python
   # Use fixed seeds when randomness is involved
   np.random.seed(42)
   torch.manual_seed(42)
   ```

### Best Practices

- Keep tests independent (no shared state)
- Use descriptive test names
- Test one thing per test
- Use fixtures for common setup
- Clean up resources (use tempfile)
- Assert meaningful conditions
- Document complex test logic

## Continuous Integration

Tests are designed to run in CI environments:

- No GPU required (CPU tests)
- Reasonable execution time
- No external dependencies beyond package requirements
- Deterministic results

## Debugging Tests

### Run with debugging output
```bash
pytest -vv --tb=long
```

### Run with print statements visible
```bash
pytest -s
```

### Run specific test with pdb
```bash
pytest --pdb test_file.py::test_function
```

### View full output on failure
```bash
pytest --tb=long --showlocals
```

## Performance Considerations

Integration tests use real models which are slower. Expected times:

- Unit tests: < 1s each
- Integration tests: 1-5s each
- End-to-end tests: 5-30s each
- Full test suite: 1-3 minutes

## Coverage

Generate coverage report:
```bash
pytest --cov=ritm_annotation --cov-report=html
open htmlcov/index.html
```

Target coverage: > 80% for core modules

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Include both unit and integration tests
3. Test edge cases and error conditions
4. Update this README if adding new test categories
5. Ensure all tests pass before committing

## Troubleshooting

### Import errors
```bash
# Install package in editable mode
pip install -e .
```

### Missing dependencies
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-timeout
```

### Model not found
Tests use lightweight models defined in fixtures. No external model files needed.

### Timeout errors
Integration tests may take longer on slow machines. Increase timeout:
```bash
pytest --timeout=600  # 10 minutes
```
