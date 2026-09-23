# Security Policy

## Project

CS2 Tactical Intelligence is a research and engineering project for
Counter-Strike 2 demo analysis, game-state processing, tactical analytics,
and machine-learning evaluation.

This document describes how to report security vulnerabilities and the
security principles used when developing and operating the project.

A security policy is not a guarantee that every safeguard described below
has already been implemented. Security controls must be enforced and tested
in the relevant code.

## Supported Versions

Security fixes are considered for the current `main` branch on a
best-effort basis.

Historical research checkpoints, archived experiments, old branches, and
previously frozen results are retained for reproducibility. They should
not be assumed to receive security updates.

There is currently no separate security-support commitment for tagged
releases or deployed services.

## Reporting a Vulnerability

**Do not disclose an unpatched vulnerability through a public GitHub issue,
pull request, discussion, or repository comment.**

If GitHub private vulnerability reporting is enabled for this repository,
use the repository's **Security → Advisories → Report a vulnerability**
workflow.

If that option is unavailable, contact the maintainer through an existing
private communication channel and request a secure reporting method.
Do not post exploit details publicly while arranging private contact.

A useful report includes:

- A concise description of the vulnerability and its potential impact.
- The affected component, commit, and relevant configuration.
- Reproduction steps or a minimal proof of concept.
- The expected behavior and the observed behavior.
- Any practical mitigation, if known.

Remove credentials, personal information, and unnecessary raw demo data
from the report. Do not conduct testing against infrastructure or accounts
that you do not own or have permission to assess.

The maintainer will review reports on a best-effort basis, assess their
impact, and coordinate an appropriate fix and disclosure timeline.
Specific response or remediation deadlines are not guaranteed.

Ordinary bugs without a security impact may be reported through the
project's regular issue tracker.

## Security-Sensitive Components

The following areas require particular care.

### Untrusted demos and archives

Downloaded Counter-Strike 2 demos, RAR/ZIP/7z archives, match-page HTML,
and archive metadata must be treated as untrusted input.

Implementations should:

- Validate the source and intended match identity before acquisition.
- Apply explicit network, disk-space, file-size, and resource limits.
- Validate archive signatures rather than trusting file extensions.
- Inspect archive members before extraction.
- Reject absolute paths, parent-directory traversal, unsafe links,
  unexpected member types, and ambiguous target members.
- Extract only the intended member into a controlled destination.
- Prevent overwriting existing research evidence or unrelated files.
- Verify recorded sizes, hashes, and the actual parsed demo identity.
- Stop for review when input identity or integrity cannot be established.

An archive filename or match-page label alone does not prove that a demo
contains the intended map, match, or tick rate.

### Network requests and local services

Network-facing components should validate URL schemes, hosts, redirects,
request sizes, and timeouts. An allowed initial URL does not automatically
authorize every redirect destination.

Any future HTTP, Game State Integration, or live telemetry endpoint must
explicitly define its allowed clients and exposure. Do not assume a local
development endpoint is safe to expose to a public network.

### Research data integrity

Frozen queues, acquisition protocols, evidence, and formal eligibility
decisions are research-integrity boundaries.

Processing code should preserve rank order, maintain traceable provenance,
verify relevant content hashes, and avoid silently replacing existing
evidence.

A technical-eligibility decision must not depend on confirmation-model
predictions or model performance.

Missing, contradictory, or unverifiable evidence requires review rather
than a fabricated result or an automatic eligibility decision.

### Secrets and sensitive information

Never commit API tokens, passwords, private keys, authentication cookies,
session headers, or private environment configuration.

Keep local secrets outside version control, use appropriately restricted
access permissions, and revoke or rotate any secret that is exposed.

Raw demo data, temporary files, debug logs, and generated artifacts should
be reviewed before publication for sensitive information and usage rights.

### Dependencies and model artifacts

Use maintained dependencies and review updates before deploying them.

Treat externally obtained Python packages, serialized models, checkpoints,
scripts, and binary tools as untrusted until their origin and intended
behavior have been assessed.

Do not deserialize untrusted pickle or joblib files or execute downloaded
scripts merely to inspect their contents.

## Development and Review Practices

Changes to security-sensitive components should include:

- Clearly defined input validation and failure behavior.
- Tests for malformed, unexpected, and adversarial inputs.
- Safe handling of interrupted operations and repeated execution.
- Explicit checks before modifying or deleting existing files.
- Review of filesystem, network, and subprocess side effects.
- Verification that secrets and large raw datasets are not accidentally
  added to version control.

Automated tests should use isolated fixtures where possible. Running
tests should not require downloading real match archives, contacting
production services, or executing confirmation-model scoring.

## Responsible Use

The project is intended for authorized research, historical replay,
observer/spectator analysis, and appropriately permitted environments.

Contributors must respect applicable game rules, platform terms,
privacy requirements, and data-source usage rights.

Do not use this project to access unauthorized game information, evade
anti-cheat systems, compromise other players, or interfere with live
competitive matches.

## Scope of This Policy

This policy covers security issues in code and services maintained in
this repository.

Issues in Counter-Strike 2, Valve infrastructure, HLTV, third-party
parsers, hosting providers, and other external systems should also be
reported through the affected provider's appropriate security channel.

Do not disclose third-party vulnerabilities publicly through this
repository.
