# -*- coding: utf-8 -*-
import os
import re
import time
import torch
from dotenv import load_dotenv
from PIL import Image

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import pipeline

#https://cloud.google.com/storage/docs/samples/storage-transfer-manager-download-bucket?hl=ko#code-sample

class Gemma3API:

    def __init__(self, model_id=None, model_type="sql"):
        # Gemma 3 모델 초기화
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not model_id:
            self.model = None
            self.tokenizer = None
            self.processor = None
        elif model_type == "sql":
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 # 메모리 사용량 감소를 위해 bfloat16 사용
            ).to(self.device)
        elif model_type == "product":
            # Load Model with PEFT adapter
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

    def load_model(self, model_id):
        """
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 # 메모리 사용량 감소를 위해 bfloat16 사용
        ).to(self.device)

    def load_product_model(self, model_id):
        """
        """
        # Load Model with PEFT adapter
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate_chat(self, prompt, max_tokens):
        """
        """

        # Hugging Face Chat Template 사용
        chat = [{"role": "user", "content": prompt}]
        prompt_template = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
        inputs = self.tokenizer(prompt_template, return_tensors="pt").to(self.device)

        # 텍스트 생성
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text
    
    def generate_sql(self, prompt, max_tokens):
        """
        """

        # Hugging Face Chat Template 사용
        # Convert as test example into a prompt with the Gemma template
        stop_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
        prompt_template = self.tokenizer.apply_chat_template(prompt["messages"][:2], tokenize=False, add_generation_prompt=True)
        #prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    
        inputs = self.tokenizer(prompt_template, return_tensors="pt").to(self.device)

        # 텍스트 생성
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text
    
    def generate_sql_pipeline(self, prompt, max_tokens):
        """
        """

        # Load the model and tokenizer into the pipeline
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        # Convert as test example into a prompt with the Gemma template
        stop_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
        prompt_template = pipe.tokenizer.apply_chat_template(prompt["messages"][:2], tokenize=False, add_generation_prompt=True)

        # Generate our SQL query.
        outputs = pipe(prompt_template, max_new_tokens=max_tokens, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=stop_token_ids, disable_compile=True)

        # Extract the user query and original answer
        print(f"Context:\n", re.search(r'<SCHEMA>\n(.*?)\n</SCHEMA>', prompt['messages'][0]['content'], re.DOTALL).group(1).strip())
        print(f"Query:\n", re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', prompt['messages'][0]['content'], re.DOTALL).group(1).strip())
        print(f"Original Answer:\n{prompt['messages'][1]['content']}")
        print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
    
        return outputs[0]['generated_text']
    
    def generate_product_description(self, product):
        product_prompt = self.prompt_product(product)
        messages = product_prompt.get("messages")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Process the image and text
        image_inputs = self.process_vision_info(messages)
        # Tokenize the text and process the images
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Move the inputs to the device
        inputs = inputs.to(self.model.device)

        # Generate the output
        stop_token_ids = [self.processor.tokenizer.eos_token_id, self.processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
        generated_ids = self.model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8, eos_token_id=stop_token_ids, disable_compile=True)
        # Trim the generation and decode the output to text
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def prompt_sql(self, data):
        # System message for the assistant
        system_message = """You are a text to SQL query translator. 
Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

        # User prompt that combines the user query and the schema
        user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

        return {
            "messages": [
                # {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt.format(question=data["sql_prompt"], context=data["sql_context"])},
                {"role": "assistant", "content": data["sql"]}
            ]
        }
    
    # Convert dataset to OAI messages
    def prompt_product(self, product):
        # System message for the assistant
        system_message = "You are an expert product description writer for Amazon."

        # User prompt that combines the user query and the schema
        user_prompt = """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
{product}
</PRODUCT>

<CATEGORY>
{category}
</CATEGORY>
"""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt.format(
                            product=product["Product Name"],
                            category=product["Category"],
                        ),
                    },
                    {
                        "type": "image",
                        "image": product["image"],
                    },
                ],
            },
        ]
        
        if "description" in product:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": product["description"]}],
            })

        return {
            "messages": messages,
        }
    
    def process_vision_info(self, messages: list[dict]) -> list[Image.Image]:
        image_inputs = []
        # Iterate through each conversation
        for msg in messages:
            # Get content (ensure it's a list)
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]

            # Check each content element for images
            for element in content:
                if isinstance(element, dict) and (
                    "image" in element or element.get("type") == "image"
                ):
                    # Get the image and convert to RGB
                    if "image" in element:
                        image = element["image"]
                    else:
                        image = element
                    image_inputs.append(image.convert("RGB"))
        return image_inputs


if __name__ == "__main__":
    # .env 파일 로드
    load_dotenv()

    start_time = time.time()
    model_id = "google/gemma-3-1b-pt"
    gemma3_api = Gemma3API(model_id)
    
    gemma3_api.generate_chat("Google Cloud Run의 장점은 무엇인가요?", 150)
    end_time = time.time()
    print(f'Elapsed time: {(end_time - start_time)}s')

    sql_context = "CREATE TABLE Donor (DonorID int, DonorName varchar(50), Country varchar(50)); INSERT INTO Donor VALUES (1, 'John Smith', 'USA'), (2, 'Jane Smith', 'Canada');"
    sql_prompt = "What is the total amount donated by each donor in the US?"
    sql = "SELECT DonorName, SUM(DonationAmount) as TotalDonated FROM Donor JOIN Donation ON Donor.DonorID = Donation.DonorID WHERE Country = 'USA' GROUP BY DonorName;"

    test_data = {"sql_context": sql_context, "sql_prompt": sql_prompt, "sql": sql}
    test_prompt = gemma3_api.prompt_sql(test_data)

    gemma3_api.generate_sql(test_prompt, 150)
    end_time2 = time.time()
    print(f'Elapsed time: {(end_time2 - end_time)}s')
    print(f'Total elapsed time: {(end_time2 - start_time)}s')

    """
    test_prompt = {
        'messages': [
            {'role': 'user', 'content': "Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.\n\n<SCHEMA>\nCREATE TABLE Donor (DonorID int, DonorName varchar(50), Country varchar(50)); INSERT INTO Donor VALUES (1, 'John Smith', 'USA'), (2, 'Jane Smith', 'Canada');\n</SCHEMA>\n\n<USER_QUERY>\nWhat is the total amount donated by each donor in the US?\n</USER_QUERY>\n"}, 
            {'role': 'assistant', 'content': "SELECT DonorName, SUM(DonationAmount) as TotalDonated FROM Donor JOIN Donation ON Donor.DonorID = Donation.DonorID WHERE Country = 'USA' GROUP BY DonorName;"}
        ]
    }
    """
    