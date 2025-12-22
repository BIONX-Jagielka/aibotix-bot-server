from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import stripe
from supabase import create_client
import logging

from app.config import settings

router = APIRouter()

logger = logging.getLogger("billing")

stripe.api_key = settings.STRIPE_SECRET_KEY

supabase = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_SERVICE_ROLE_KEY,
)


class CreateCheckoutSessionRequest(BaseModel):
    user_id: str
    email: str | None = None


@router.post("/create-checkout-session")
async def create_checkout_session(payload: CreateCheckoutSessionRequest):
    try:
        user = supabase.table("users").select("id").eq("id", payload.user_id).single().execute()
        if not user.data:
            raise HTTPException(status_code=404, detail="User not found")

        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[
                {
                    "price": settings.STRIPE_LIVE_PRICE_ID,
                    "quantity": 1,
                }
            ],
            success_url=settings.FRONTEND_SUCCESS_URL,
            cancel_url=settings.FRONTEND_CANCEL_URL,
            customer_email=payload.email if payload.email else None,
            subscription_data={
                "metadata": {
                    "user_id": payload.user_id,
                    "product": "AIBOTIX_LIVE",
                }
            },
        )

        supabase.table("subscriptions_pending").insert({
            "user_id": payload.user_id,
            "stripe_session_id": session.id,
            "price_id": settings.STRIPE_LIVE_PRICE_ID,
            "status": "created",
        }).execute()

        return {"checkout_url": session.url}

    except Exception as e:
        logger.exception("Failed to create checkout session")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")