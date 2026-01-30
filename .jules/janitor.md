# Janitor's Journal

This journal is a record of critical learnings related to code quality and refactoring in this repository. Its purpose is to help maintain consistency and avoid repeating past mistakes.

## 2025-12-31 - Remove Commented-Out Code
**Issue:** The file `ritm_annotation/cli/annotate/annotator.py` contained several blocks of commented-out code.
**Root Cause:** The commented code appeared to be remnants of previous debugging or experimental code that was no longer needed but was never removed.
**Solution:** I removed the unused, commented-out code blocks to clean up the file and improve readability.
**Pattern:** Periodically scan for and remove dead or commented-out code. Such code adds clutter, can be confusing to new developers, and provides no functional value. Keeping the codebase clean of such artifacts is a simple but effective maintenance practice.

## 2026-01-30 - Remove Unused Imports in Tests
**Issue:** The file `ritm_annotation/utils/test_misc.py` imported `itertools` and `pytest` but never used them.
**Root Cause:** These imports were likely left over from previous test iterations or copied from other test files without being cleaned up.
**Solution:** Removed the unused imports to reduce clutter and avoid confusion.
**Pattern:** Always check for and remove unused imports, especially in test files where they can accumulate unnoticed. Clean imports make dependencies clear.
