# aibotix/utils/http_client.py
from __future__ import annotations
import random
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("aibotix")

@dataclass
class HttpConfig:
    connect_timeout: float = 3.0
    read_timeout: float = 6.0
    total_timeout_cap: float = 12.0
    max_retries: int = 5
    backoff_base: float = 0.4
    backoff_cap: float = 4.0

class ResilientSession:
    def __init__(self, cfg: HttpConfig | None = None):
        self.cfg = cfg or HttpConfig()
        self.session = requests.Session()
        retry = Retry(
            total=self.cfg.max_retries,
            connect=self.cfg.max_retries,
            read=self.cfg.max_retries,
            status=self.cfg.max_retries,
            backoff_factor=0.0,  # we do our own jittered backoff
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET","POST","PUT","DELETE"),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def request(self, method: str, url: str, *, headers: Optional[Dict[str,str]]=None,
                params: Optional[Dict[str,Any]]=None, json: Any=None, context: Optional[Dict[str,Any]]=None):
        ctx = context or {}
        t0 = time.time()
        attempt = 0
        last_exc = None
        while True:
            attempt += 1
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    timeout=(self.cfg.connect_timeout, self.cfg.read_timeout),
                )
                # Hard fail on 4xx except 429 (retry handles status codes but we still gate)
                if resp.status_code >= 400 and resp.status_code not in (429,):
                    log.warning("[HTTP] non_retryable_status", extra={"status": resp.status_code, "url": url, **ctx})
                    return resp
                return resp
            except Exception as e:
                last_exc = e
                elapsed = time.time() - t0
                log.warning("[HTTP] exception", extra={"attempt": attempt, "elapsed": round(elapsed,3), "url": url, "exc": type(e).__name__, **ctx})
                if elapsed >= self.cfg.total_timeout_cap or attempt >= self.cfg.max_retries:
                    raise
                sleep = min(self.cfg.backoff_cap, self.cfg.backoff_base * (2 ** (attempt-1)))
                sleep = sleep * (0.8 + 0.4 * random.random())  # jitter
                time.sleep(sleep)