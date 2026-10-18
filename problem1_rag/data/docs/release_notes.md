# Zentra Cloud Release Notes

## v2.4.0 (March 12, 2025)

Added automatic webhook retries with exponential delay. Added dark mode to the web dashboard.

## v2.3.0 (January 20, 2025)

Introduced SAML single sign-on for the Team plan. The rate limit of the Pro plan was raised from 300 to 600 requests per minute.

## v2.2.1 (December 5, 2024)

Security patch that fixes a token expiry bug. Fixed the character encoding of CSV exports.

## v2.2.0 (November 14, 2024)

Cursor-based pagination replaced offset pagination. Offset pagination is deprecated and will be removed on September 30, 2025.

## v2.1.0 (September 3, 2024)

Released the Python SDK 1.0 and the Node SDK 1.0.

## Deprecation schedule

API v1 will be shut down on December 31, 2025. Customers still using v1 should migrate to v2 before that date.

## v2.0.0 (June 18, 2024)

Launched the v2 API with a new authentication model based on Bearer tokens. Added the sandbox environment.

## v1.9.2 (April 2, 2024)

Fixed a bug where webhook deliveries could be sent twice. Improved the error message returned for expired API keys.

## v1.9.0 (February 6, 2024)

Added the audit log viewer to the dashboard. Increased the maximum single file upload size from 50 MB to 100 MB.

## Upcoming

The Go SDK is expected to leave beta in the second half of 2025. A mobile dashboard application is on the roadmap but has no release date yet.
