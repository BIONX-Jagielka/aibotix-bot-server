import os
import stripe

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_LIVE_PRICE_ID")
FRONTEND_URL = os.getenv("FRONTEND_URL")

stripe.api_key = STRIPE_SECRET_KEY


def create_checkout_session(user_id: str, email: str):
    session = stripe.checkout.Session.create(
        mode="subscription",
        payment_method_types=["card"],
        customer_email=email,
        line_items=[
            {
                "price": STRIPE_PRICE_ID,
                "quantity": 1,
            }
        ],
        metadata={
            "user_id": user_id
        },
        success_url=f"{FRONTEND_URL}/billing/success",
        cancel_url=f"{FRONTEND_URL}/billing/cancel",
    )

    return session.url