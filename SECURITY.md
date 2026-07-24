# Security Policy

## Reporting a Vulnerability

Please do not open a public issue for a vulnerability that could expose API
keys, user documents, or deployment details. Contact the repository owner
privately with:

- A description of the issue
- Reproduction steps
- The affected files or versions
- The potential impact

## Credential Handling

- Never commit `.env` or `.streamlit/secrets.toml`.
- Prefer a managed secret store for deployed environments.
- Rotate any credential that has appeared in Git history or logs.
- Use separate development and production deployments where possible.

## Document Privacy

Uploaded documents are processed in application memory. Text sent to an Azure
embedding or generation deployment leaves the local process and is governed by
that deployment's data and network policies.
