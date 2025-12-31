# Janitor's Journal

This journal is a record of critical learnings related to code quality and refactoring in this repository. Its purpose is to help maintain consistency and avoid repeating past mistakes.

## 2025-12-31 - Remove Commented-Out Code
**Issue:** The file `ritm_annotation/cli/annotate/annotator.py` contained several blocks of commented-out code.
**Root Cause:** The commented code appeared to be remnants of previous debugging or experimental code that was no longer needed but was never removed.
**Solution:** I removed the unused, commented-out code blocks to clean up the file and improve readability.
**Pattern:** Periodically scan for and remove dead or commented-out code. Such code adds clutter, can be confusing to new developers, and provides no functional value. Keeping the codebase clean of such artifacts is a simple but effective maintenance practice.
