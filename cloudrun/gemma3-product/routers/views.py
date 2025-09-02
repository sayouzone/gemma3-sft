from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder

router = APIRouter(
  prefix="/view",
  tags=['views'],
  responses={404: {"description": "Not found"}}
)