import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # =========================
    # ENVIRONMENT
    # =========================
    ENV: str = "development"

    # =========================
    # STRIPE CONFIG
    # =========================
    STRIPE_SECRET_KEY: str
    STRIPE_PUBLISHABLE_KEY: str
    STRIPE_WEBHOOK_SECRET: str

    # Stripe price ID for AIBOTIX LIVE (£10/month)
    STRIPE_LIVE_PRICE_ID: str

    # =========================
    # SUPABASE CONFIG
    # =========================
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str

    # =========================
    # FRONTEND URLS
    # =========================
    FRONTEND_SUCCESS_URL: str
    FRONTEND_CANCEL_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()