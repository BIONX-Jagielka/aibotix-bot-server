from fastapi import APIRouter
import stripe
from app.config import settings
from pydantic import BaseModel
from fastapi import HTTPException

router = APIRouter()

stripe.api_key = settings.STRIPE_SECRET_KEY


class CheckoutRequest(BaseModel):
    user_id: str


@router.post("/stripe/create-checkout-session")
async def create_checkout_session(payload: CheckoutRequest):
    try:
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
            metadata={
                "user_id": payload.user_id,
                "product": "AIBOTIX_LIVE"
            }
        )

        return {"checkout_url": session.url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))