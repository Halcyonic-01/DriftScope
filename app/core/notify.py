"""
app/core/notify.py

One place that knows how to put an email on the wire.

Two callers with different urgencies share it:

  * app/core/canary.py sends an *alert* — something is wrong, drift
    crossed the threshold.
  * scripts/notify_run.py sends a *heartbeat* — a scheduled job finished,
    here is what it did. GitHub only emails you when a workflow fails, so
    without this a run that silently stopped producing useful results
    looks exactly like a healthy one from the inbox.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def smtp_is_configured() -> bool:
    """True when every setting needed to send mail is present."""
    from app.core.config import settings

    return all(
        [
            settings.alert_email,
            settings.smtp_host,
            settings.smtp_port,
            settings.smtp_user,
            settings.smtp_password,
        ]
    )


def send_email(subject: str, body: str, to_email: str | None = None) -> bool:
    """
    Send a plain-text email using the configured SMTP settings.

    Returns True if it went out, False if SMTP isn't configured. Delivery
    failures raise — callers that must not fail their job because of the
    mail server (the run-summary notifier) catch them.
    """
    from app.core.config import settings

    if not smtp_is_configured():
        logger.warning("Email not sent (%s): SMTP settings are incomplete", subject)
        return False

    recipient = to_email or settings.alert_email
    send_email_via(
        subject=subject,
        body=body,
        to_email=recipient,
        smtp_host=settings.smtp_host,
        smtp_port=settings.smtp_port,
        smtp_user=settings.smtp_user,
        smtp_password=settings.smtp_password,
    )
    return True


def send_email_via(
    subject: str,
    body: str,
    to_email: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
) -> None:
    """Explicit-credentials variant, so tests can drive it without settings."""
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_email, msg.as_string())
