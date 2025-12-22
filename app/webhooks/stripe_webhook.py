import os
import stripe
from fastapi import APIRouter, Request, HTTPException
from supabase import create_client, Client
from datetime import datetime, timezone

router = APIRouter()

# Environment variables
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Safety checks
if not all([STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise RuntimeError("Missing required environment variables for Stripe webhook")

stripe.api_key = STRIPE_SECRET_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


@router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook payload")

    event_type = event["type"]
    data = event["data"]["object"]

    # 1️⃣ Successful checkout → activate subscription
    if event_type == "checkout.session.completed":
        user_id = data.get("metadata", {}).get("user_id")
        subscription_id = data.get("subscription")
        customer_id = data.get("customer")

        if user_id and subscription_id and customer_id:
            supabase.table("user_subscriptions").upsert({
                "user_id": user_id,
                "stripe_subscription_id": subscription_id,
                "stripe_customer_id": customer_id,
                "status": "active",
                "plan": "live",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, on_conflict="user_id").execute()

    # 2️⃣ Payment failed → mark as past_due
    elif event_type == "invoice.payment_failed":
        subscription_id = data.get("subscription")

        if subscription_id:
            supabase.table("user_subscriptions").update({
                "status": "past_due",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("stripe_subscription_id", subscription_id).execute()

    elif event_type == "customer.subscription.updated":
        subscription_id = data.get("id")
        status = data.get("status")

        if subscription_id and status:
            supabase.table("user_subscriptions").update({
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("stripe_subscription_id", subscription_id).execute()

    # 3️⃣ Subscription cancelled → disable access
    elif event_type == "customer.subscription.deleted":
        subscription_id = data.get("id")

        if subscription_id:
            supabase.table("user_subscriptions").update({
                "status": "canceled",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("stripe_subscription_id", subscription_id).execute()

    else:
        # Unhandled but acknowledged event
        pass

    # Always return 200 to Stripe
    return {"received": True}
