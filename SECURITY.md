# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Darwin Scaffold Studio, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly at: demetrios@agourakis.med.br
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution**: Depends on severity (critical: 1 week, high: 2 weeks, medium: 1 month)

## Security Considerations

### Data Handling

- Darwin Scaffold Studio processes medical imaging data (MicroCT, SEM)
- All processing is done locally; no data is transmitted externally
- Users are responsible for compliance with their institution's data policies

### Dependencies

- Dependencies are pinned in `Manifest.toml` for reproducibility
- Regular dependency audits are performed
- Update to latest compatible versions when security patches are released

### File System Access

- The software reads/writes files only in user-specified directories
- No system-wide file access is required
- Docker containers provide additional isolation

## Best Practices for Users

1. **Keep Updated**: Use the latest release
2. **Verify Downloads**: Check release signatures when available
3. **Isolate Sensitive Data**: Use Docker for processing sensitive datasets
4. **Review Outputs**: Verify exported data before sharing

## Scope

This security policy covers:
- The Darwin Scaffold Studio Julia package
- Docker configurations in this repository
- GitHub Actions workflows

It does NOT cover:
- Third-party dependencies (report to their maintainers)
- User-created extensions or modifications
- Deployment infrastructure (user's responsibility)
