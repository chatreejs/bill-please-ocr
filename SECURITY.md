# Security Policy

## Introduction

This document outlines the security policy for Bill Please OCR, the receipt OCR service behind [Bill Please API](https://github.com/chatreejs/bill-please-api) for the [Bill Please](https://bill-please.chatree.dev/) open source project. The policy is intended to provide guidelines and procedures for reporting, triaging, and addressing security vulnerabilities in the project.

## Scope

The security policy covers the codebase, Docker image and documentation of this repository, including:

- The OCR endpoint (`POST /v1/ocr`) and the probes (`GET /healthz`, `GET /readyz`).
- Handling of uploaded receipt images (type and size limits, decoding with Pillow and pi-heif, EXIF rotation and downscaling).
- Logging, which must never contain receipt images or recognized text.
- The Docker image, including the ONNX models baked into it.

Vulnerabilities in the receipt scan API should be reported according to the [Bill Please API security policy](https://github.com/chatreejs/bill-please-api/blob/main/SECURITY.md), and vulnerabilities in the Bill Please web app according to the [Bill Please security policy](https://github.com/chatreejs/bill-please/blob/main/SECURITY.md). Vulnerabilities in third-party dependencies (ONNX Runtime, OpenCV, FastAPI, Pillow, pi-heif) should be reported to their maintainers; let us know too if this project is affected.

## Supported Versions

Only the latest release on the `main` branch receives security fixes.

## Vulnerability Disclosure Process

**Please do not report security vulnerabilities through public GitHub issues, discussions or pull requests.**

Send vulnerability reports to the dedicated email address **contact@chatree.dev**, or use GitHub's [private vulnerability reporting](https://github.com/chatreejs/bill-please-ocr/security/advisories/new) if it is enabled. Please include:

- A description of the issue and its impact.
- Steps to reproduce (requests, crafted images, configuration) or a proof of concept. Never send real receipts.
- The affected version or commit.

Vulnerability reports will be reviewed and triaged by the project's maintainers. The owner will aim to respond to vulnerability reports within 72 hours and will provide regular updates on the status of the vulnerability and any remediation efforts.

## Roles and Responsibilities

The maintainers are responsible for handling vulnerability reports and making decisions about how to address them. They will also work with contributors and reporters to resolve the issue(s) as quickly as possible.

## Response Timeline

Bill Please OCR will aim to resolve critical vulnerabilities within 30 days and non-critical vulnerabilities within 90 days. These deadlines may extend if additional time is needed to address the issue(s).

## Secure Deployment Notes

If you host your own instance:

- The service has **no authentication** and is meant to be called only by bill-please-api. Do not expose it publicly; restrict it to the api pods, for example with a Kubernetes NetworkPolicy.
- Keep `MAX_IMAGE_BYTES` and `MAX_IMAGE_SIDE` low. OCR runs one request at a time on a shared GPU, so large or many images can starve the service.
- Run the container as a non-root user, as the published image does (uid 1000).
- Keep `LOG_LEVEL` at `INFO` or above in production, and never add logging of images or recognized text: receipts can hold names and card digits.

## Secure Coding Practices

Contributors should validate all input (MIME type, size, decoded dimensions), never log receipt images, recognized text or secrets, never commit real receipts, and keep dependencies up to date.

## Regular Review and Update

The security policy will be regularly reviewed and updated to ensure that it remains effective and relevant. The maintainers will evaluate the vulnerability disclosure process, update secure coding guidelines, and revise the response timeline as needed.

## Disclosure Policy

Bill Please OCR will follow a coordinated disclosure policy, which means that vulnerabilities will be disclosed publicly only after they have been remediated. Reporters will be credited unless they prefer to remain anonymous.

## Legal Disclaimer

The software is provided "as is", without warranty of any kind, as described in the [AGPL-3.0 License](./LICENSE). The project maintainers and contributors are not liable for any security vulnerabilities or incidents that occur as a result of using or self-hosting Bill Please OCR.

## Contact Information

If you have any questions or concerns about the security policy or any security vulnerabilities in the project, please contact us at _contact@chatree.dev_.

By implementing this security policy, we aim to ensure that vulnerabilities are addressed promptly and that users and contributors can use Bill Please OCR and its linked sources safely and securely.
