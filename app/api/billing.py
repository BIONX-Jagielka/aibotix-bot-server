from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import stripe
from supabase import create_client
import logging

from app.config import settings

router = APIRouter(prefix="/billing", tags=["billing"])

logger = logging.getLogger("billing")

stripe.api_key = settings.STRIPE_SECRET_KEY

supabase = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_SERVICE_ROLE_KEY,
)


class CreateCheckoutSessionRequest(BaseModel):
    user_id: str
    email: str | None = None


class CreatePortalSessionRequest(BaseModel):
    user_id: str


# New request model for cancel subscription
class CancelSubscriptionRequest(BaseModel):
    user_id: str


@router.post("/create-checkout-session")
async def create_checkout_session(payload: CreateCheckoutSessionRequest):
    try:
        # Get or create Stripe customer
        customer = None

        if payload.email:
            customers = stripe.Customer.list(email=payload.email, limit=1)
            if customers.data:
                customer = customers.data[0]
            else:
                customer = stripe.Customer.create(
                    email=payload.email,
                    metadata={"user_id": payload.user_id}
                )

        # Create Stripe checkout session
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            customer=customer.id if customer else None,
            line_items=[
                {
                    "price": settings.STRIPE_LIVE_PRICE_ID,
                    "quantity": 1,
                }
            ],
            success_url=settings.FRONTEND_SUCCESS_URL,
            cancel_url=settings.FRONTEND_CANCEL_URL,
            subscription_data={
                "metadata": {
                    "user_id": payload.user_id,
                    "product": "AIBOTIX_LIVE",
                }
            },
        )

        return {"url": session.url}

    except Exception as e:
        logger.exception("Stripe checkout session creation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-portal-session")
async def create_portal_session(payload: CreatePortalSessionRequest):
    try:
        # Fetch user's Stripe customer ID from Supabase
        result = (
            supabase
            .table("user_subscriptions")
            .select("stripe_customer_id")
            .eq("user_id", payload.user_id)
            .single()
            .execute()
        )

        stripe_customer_id = result.data.get("stripe_customer_id")

        if not stripe_customer_id:
            raise HTTPException(status_code=400, detail="Stripe customer not found")

        # Create Stripe customer portal session
        portal_session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=settings.FRONTEND_SUCCESS_URL,
        )

        return {"portal_url": portal_session.url}

    except Exception as e:
        logger.exception("Stripe portal session creation failed")
        raise HTTPException(status_code=500, detail=str(e))


# New route for cancelling a subscription
@router.post("/cancel-subscription")
async def cancel_subscription(payload: CancelSubscriptionRequest):
    try:
        # Fetch user's Stripe subscription ID from Supabase
        result = (
            supabase
            .table("user_subscriptions")
            .select("stripe_subscription_id")
            .eq("user_id", payload.user_id)
            .single()
            .execute()
        )

        stripe_subscription_id = result.data.get("stripe_subscription_id")

        if not stripe_subscription_id:
            raise HTTPException(status_code=400, detail="Stripe subscription not found")

        # Cancel subscription immediately
        stripe.Subscription.delete(stripe_subscription_id)

        return {"status": "cancelled"}

    except Exception as e:
        logger.exception("Stripe subscription cancellation failed")
        raise HTTPException(status_code=500, detail=str(e))