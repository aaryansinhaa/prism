"""Health and status endpoints."""

from typing import Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def read_root() -> Dict[str, str]:
    return {"service": "prism-runtime"}
