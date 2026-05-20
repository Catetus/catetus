# Security Policy

## Reporting a Vulnerability

If you believe you have found a security issue in Catetus — anything from
arbitrary code execution in the viewer / CLI, to a sandbox escape in the
asset pipeline, to a credential-leak vector in any of our standards-outreach
tooling — please **do not** open a public GitHub issue.

Instead, either:

- Open a private report via GitHub Security Advisories:
  <https://github.com/Catetus/catetus/security/advisories/new>, **or**
- Email **security@catetus.com** with a description, reproduction steps,
  and any proof-of-concept material.

We aim to acknowledge new reports within **7 days**. We do not currently
offer a bug bounty.

Please give us a reasonable window to investigate and ship a fix before any
public disclosure.

## Supported Versions

Catetus is pre-1.0. Security fixes are applied to the latest `0.x`
release line only.

| Version | Supported          |
| ------- | ------------------ |
| 0.x     | :white_check_mark: |

## Scope

In-scope:

- The Rust workspace (`crates/*`) — CLI, optimize pipeline, codec, viewer
  glue, validators.
- The TypeScript packages (`packages/*`) — viewer, website, sidecar tooling.
- The container/worker images used by `apps/optimize-action` and the
  hosted `apps/web` / Astro site.

Out-of-scope:

- Third-party Gaussian Splat scenes loaded into the viewer; we treat input
  scenes as untrusted by design.
- Social-engineering or physical-access issues against contributors.
- Best-practice findings without a demonstrated impact (e.g. "you should
  use header X") — open a normal issue or PR.

Thank you for helping keep Catetus users safe.
