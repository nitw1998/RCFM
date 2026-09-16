# Security and private research disclosure

Do not open a public issue containing unpublished methods, private datasets,
patient information, credentials, checkpoints, or private experiment configs.
Contact the repository owner privately through the address associated with the
submitted manuscript.

Unpublished research should be developed in a separate private repository. The
public `main` branch accepts only the RCFM-OT configuration scope enforced by
`scripts/audit_public_release.py`. CODEOWNERS and the release-guard workflow are
defense-in-depth checks; repository administrators should also require pull
requests and owner approval in GitHub branch-protection settings.
