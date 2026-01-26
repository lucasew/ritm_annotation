## 2026-01-26 - Fix insecure dynamic class loading

**Vulnerability:** The function `ritm_annotation.utils.serialization.get_class_from_str` allowed loading and instantiating arbitrary classes from any installed module (e.g., `subprocess.Popen`) based on a string input. This could allow an attacker to achieve Arbitrary Code Execution (ACE) by providing a malicious configuration or model file.

**Learning:** Dynamic class loading and instantiation based on user-controlled strings is a common vector for RCE/ACE. It allows attackers to bypass intended application logic and interact directly with the python runtime and system resources.

**Prevention:** Always restrict dynamic loading to a strict allowlist of safe, expected namespaces. In this case, we restricted loading to `ritm_annotation.`, `isegm.`, `torch.`, and `torchvision.`.
