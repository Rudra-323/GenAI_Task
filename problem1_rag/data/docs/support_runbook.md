# Zentra Support Runbook

This runbook is for support and on-call engineers.

## Severity levels and response targets

Sev1 (critical outage): respond within 15 minutes, 24/7. Sev2 (major degradation): respond within 1 hour. Sev3 (minor issue): respond within 1 business day. Sev4 (general question): respond within 3 business days.

## Escalation

If a Sev1 ticket is not acknowledged within 15 minutes, page the on-call engineering manager. If it is still unresolved after 1 hour, escalate to the VP of Engineering.

## On-call rotation

On-call shifts last one week and the handover happens on Mondays at 10:00 UTC. Engineers on call receive an allowance of $250 per week.

## Status page

Status page updates are posted every 30 minutes during Sev1 incidents. Updates must be written in plain language without internal system names.

## Postmortems

A blameless postmortem is required for every Sev1 and Sev2 incident and must be published within 5 business days.

## Support hours

Standard support is available Monday to Friday from 08:00 to 20:00 UTC. Team plan customers can reach priority support around the clock.

## Refund approvals

Support agents may approve refunds up to $500. Refunds above $500 need approval from the finance team.

## Data requests

Customer data export requests under GDPR must be fulfilled within 30 days.

## Ticket tags

Every ticket must carry one product tag (billing, api, dashboard or security) and one severity tag. Tickets without both tags are returned to the triage queue by the daily audit.

## Language support

Support is offered in English, Spanish, German and Japanese. Tickets in other languages are answered in English with a machine-translated copy.

## Customer communication

Replies must be written in a friendly and professional tone. Agents must never promise a fix date that engineering has not confirmed. Use the approved incident template for any message about an outage.

## Abusive customers

Agents may end a conversation after one clear warning if a customer uses abusive language. The ticket must be flagged for the team lead, who decides whether the account needs a formal notice.

## Knowledge base

Knowledge base articles are reviewed every quarter by the support team. An article that has not been reviewed in 6 months is hidden from customers until it is checked again.

## Handover notes

At each on-call handover, the outgoing engineer writes a summary of open incidents, pending deployments and known risks in the shared handover document.
