# Zentra Cloud API Guide (v2)

The Zentra REST API is served from https://api.zentra.example/v2 and returns JSON for every endpoint.

## Authentication

Send your API key as a Bearer token in the Authorization header. Live keys start with zk_live_ and test keys start with zk_test_. API keys expire after 90 days unless they are rotated. Rotate keys from the dashboard under Settings, then API Keys.

## Rate limits

Rate limits depend on your plan. The Free plan allows 60 requests per minute. The Pro plan allows 600 requests per minute. The Team plan allows 3,000 requests per minute. Exceeding the limit returns HTTP 429 with a Retry-After header telling you how many seconds to wait.

## Pagination

List endpoints use cursor-based pagination. Pass the limit and cursor query parameters. The default page size is 25 and the maximum is 100. Each response includes a next_cursor field, which is null on the last page.

## Error codes

The API uses standard HTTP status codes. 400 means invalid parameters, 401 means a missing or invalid API key, 403 means insufficient permissions, 404 means the resource was not found, 409 means a conflict such as a reused idempotency key, 429 means the rate limit was exceeded, and 500 means an internal error. For 500 errors, retry with exponential backoff.

## Idempotency

Send an Idempotency-Key header on POST requests to make retries safe. Zentra stores idempotency keys for 24 hours. Reusing a key with a different request body returns HTTP 409.

## Webhooks

Webhook payloads are signed with HMAC-SHA256 and the signature is sent in the X-Zentra-Signature header. Failed deliveries are retried up to 5 times over 24 hours. The maximum webhook payload size is 256 KB.

## File uploads

A single upload request can carry a file of up to 100 MB. For larger files, use multipart upload, which supports files of up to 5 GB with a minimum part size of 8 MB.

## Versioning

The current API version is v2. Breaking changes are only introduced in a new major version, and each old version is supported for at least 12 months after its successor is released.

## SDKs

Official SDKs are available for Python and Node.js. A Go SDK is available as a beta release and is not covered by the support SLA.

## Filtering and sorting

List endpoints accept filters in the form filter[field]=value. Sorting uses the sort parameter, and a leading minus sign reverses the order, for example sort=-created_at for newest first.

## Request IDs

Every response carries an X-Request-Id header. Include this value when you contact support about a failing request so the engineers can find it in the logs.

## Timeouts

The server times out a request after 30 seconds. Operations that take longer, such as large exports, are handled as asynchronous jobs that you poll for completion.

## Browser access

Calling the API directly from a browser is allowed only with publishable keys, which start with zk_pub_. Publishable keys are read-only and cannot modify data.

## Sandbox

A sandbox environment is available at sandbox.zentra.example. Its data is reset every Sunday at 00:00 UTC, so do not store anything there that you want to keep.
