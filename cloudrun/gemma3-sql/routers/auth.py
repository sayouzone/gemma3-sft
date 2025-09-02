from fastapi import Depends, HTTPException, status, APIRouter, Request, Response, Form

router = APIRouter(
  prefix="/auth",
  tags=["auth"],
  responses={401: {"user": "Not authorized"}}
)