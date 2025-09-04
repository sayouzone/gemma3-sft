from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

import argparse
import datetime
import os
import requests
import time
from PIL import Image

from utils import gemma3api

router = APIRouter(
  prefix="",
  tags=['apis'],
  responses={404: {"description": "Not found"}}
)

@router.post("/generate", response_class=JSONResponse)
async def generate(request: Request):
    """텍스트 생성을 위한 엔드포인트"""

    generated_text = None

    json_body = await request.json()
    type = json_body.get("type", "text")
    prompt = json_body.get("prompt")
    max_tokens = json_body.get("max_tokens", 100)

    gemma3_api = request.app.state.gemma3_api

    if not prompt:
        return {"error": "Prompt is required"}, 400
    
    if type == "text":
        generated_text = gemma3_api.generate_chat(prompt, max_tokens)
    elif type == "sql":
        sql_context = "CREATE TABLE Donor (DonorID int, DonorName varchar(50), Country varchar(50)); INSERT INTO Donor VALUES (1, 'John Smith', 'USA'), (2, 'Jane Smith', 'Canada');"
        sql_prompt = "What is the total amount donated by each donor in the US?"
        sql = "SELECT DonorName, SUM(DonationAmount) as TotalDonated FROM Donor JOIN Donation ON Donor.DonorID = Donation.DonorID WHERE Country = 'USA' GROUP BY DonorName;"

        test_data = {"sql_context": sql_context, "sql_prompt": sql_prompt, "sql": sql}
        prompt = gemma3_api.prompt_sql(test_data)

        #generated_text = gemma3_api.generate_sql(prompt, max_tokens)
        generated_text = gemma3_api.generate_sql_pipeline(prompt, max_tokens)
    elif type == "product":
        # Test sample with Product Name, Category and Image
        """
        product = {
            "product_name": "Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur",
            "category": "Toys & Games | Toy Figures & Playsets | Action Figures",
            "image": Image.open(requests.get("https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg", stream=True).raw).convert("RGB")
        }
        """
        product = {
            "Product Name": "Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur",
            "Category": "Toys & Games | Toy Figures & Playsets | Action Figures",
            "image": Image.open(requests.get("https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg", stream=True).raw).convert("RGB")
        }

        # generate the description
        generated_text = gemma3_api.generate_product_description(product)
        print(generated_text)

    #return {"response": generated_text}
    return JSONResponse(content=jsonable_encoder(generated_text))
