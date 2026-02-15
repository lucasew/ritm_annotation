## IGNORE: Restricting Dynamic Class Loading

**- Pattern:** Modifying `ritm_annotation/utils/serialization.py` to restrict `get_class_from_str` with an allowlist or whitelist.
**- Justification:** Multiple attempts (PRs 138, 122, 121, 111) to restrict this function for security reasons were consistently rejected. The flexibility is likely required.
**- Files Affected:** ritm_annotation/utils/serialization.py

## IGNORE: Enforcing weights_only=True in torch.load

**- Pattern:** Adding `weights_only=True` argument to `torch.load` calls.
**- Justification:** PR 120 was rejected. While a security best practice, it likely breaks compatibility with existing models or workflows in this project.
**- Files Affected:** ritm_annotation/engine/trainer.py, ritm_annotation/inference/utils.py, ritm_annotation/model/modeling/hrnet_ocr.py

## IGNORE: Replacing Pickle with JSON

**- Pattern:** Refactoring data persistence from `pickle` to `json`.
**- Justification:** PR 110 attempting this in `sbd.py` was rejected.
**- Files Affected:** ritm_annotation/data/datasets/sbd.py

## IGNORE: Automated Dependency Bumps

**- Pattern:** Creating PRs solely to bump versions of GitHub Actions or other dependencies.
**- Justification:** Multiple PRs (134, 125, 123, 117) were closed without merge.
**- Files Affected:** .github/workflows/*.yml

## IGNORE: Removing Dead Code or Unused Imports

**- Pattern:** Removing commented-out code or unused imports as a cleanup task.
**- Justification:** Janitor PRs 130 and 129 were rejected. Maintainers may prefer keeping these artifacts for development contexts.
**- Files Affected:** ritm_annotation/utils/*.py

## IGNORE: Overhauling CI and Tooling Configuration

**- Pattern:** Massive rewrites of CI workflows and `mise.toml` to enforce "standard" tooling.
**- Justification:** Arrumador PRs 135 and 127 were rejected.
**- Files Affected:** .github/workflows/autorelease.yml, mise.toml
