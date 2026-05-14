## Summary

<!-- One-paragraph description of what this PR does. -->

## Related spec / issue

- spec: `specs/####-...md` (or "n/a — refactor")
- closes #

## Checklist

- [ ] Tests added (unit / integration / snapshot / property as appropriate)
- [ ] `cargo fmt --all -- --check` is clean
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean
- [ ] `cargo test --workspace` passes
- [ ] `pnpm -r run lint && pnpm -r run test` passes
- [ ] Public API changes documented in CHANGELOG.md under `## [Unreleased]`
- [ ] If parser/format change: malformed fixture added in `fixtures/invalid/`
- [ ] If optimizer change: before/after pass stats emitted
- [ ] If renderer change: visual-regression snapshot updated only if spec required
- [ ] Commits are `Signed-off-by:` (DCO)

## Notes for the reviewer

<!-- Anything tricky, anything you're unsure about, anything you punted on. -->
