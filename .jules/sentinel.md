## 2026-01-23 - Fix insecure deserialization in torch.load

**Vulnerability:** Found multiple instances of `torch.load` being used without `weights_only=True`. `torch.load` uses `pickle` internally, which allows arbitrary code execution if the loaded file is malicious.
**Learning:** Machine learning models are often shared as pickled files, but this format is inherently insecure. PyTorch recently added `weights_only=True` to mitigate this by restricting unpickling to safe types.
**Prevention:** Always use `weights_only=True` when using `torch.load` unless there is a specific need to load arbitrary Python objects (which should be avoided if possible).
