# Read-only application validation

The disabled managed auto-merge workflow is unchanged. This independent workflow has contents:read only, pinned actions, nonpersisted checkout credentials, no secrets, merge/deploy operations or artifact/cache uploads. Standard ubuntu-latest is used without runner registration or billing changes.

Local equivalent validation passed: 2 actual synthetic upload/analysis/Word/download tests. Existing tests were retained. Node24 or the declared Python3.11.15 environment is used. No real production data or external application services are accessed. Only ordinary dependency installation is required; automatic npm audit submission is disabled. Hosted CI has not yet been confirmed.
