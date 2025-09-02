from datasets import load_dataset

"""
Source:
https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora?hl=ko

Dataset:
philschmid/gretel-synthetic-text-to-sql
https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql
"""

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

def create_conversation(sample):
    return {
        "messages": [
            # {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
            {"role": "assistant", "content": sample["sql"]}
        ]
    }

# Load dataset from the hub
dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train")
dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
# split dataset into 10,000 training samples and 2,500 test samples
dataset = dataset.train_test_split(test_size=2500/12500)

# Print formatted user prompt
print(dataset["train"][345]["messages"][1]["content"])

print(dataset)

"""
README.md: 100%|██████████████████████████████████████████████████████| 737/737 [00:00<00:00, 3.19MB/s]
(…)nthetic_text_to_sql_train.snappy.parquet: 100%|████████████████| 32.4M/32.4M [00:02<00:00, 13.4MB/s]
(…)ynthetic_text_to_sql_test.snappy.parquet: 100%|████████████████| 1.90M/1.90M [00:00<00:00, 2.43MB/s]
Generating train split: 100%|███████████████████████| 100000/100000 [00:00<00:00, 427209.89 examples/s]
Generating test split: 100%|████████████████████████████| 5851/5851 [00:00<00:00, 977957.79 examples/s]
Map: 100%|█████████████████████████████████████████████| 12500/12500 [00:00<00:00, 27898.74 examples/s]
SELECT COUNT(image_id) FROM image_data WHERE image_date >= NOW() - INTERVAL '1 month';
"""

"""
Map: 100%|█████████████████████████████████████████████| 12500/12500 [00:00<00:00, 26551.63 examples/s]
SELECT s.supplier_name, COUNT(DISTINCT bts.brand_id) as sustainable_brand_count FROM Textile_Suppliers s JOIN Brands_Textile_Suppliers bts ON s.supplier_id = bts.supplier_id JOIN Brands b ON bts.brand_id = b.brand_id WHERE s.is_sustainable = TRUE AND b.country = 'Germany' GROUP BY s.supplier_name ORDER BY sustainable_brand_count DESC LIMIT 5;
DatasetDict({
    train: Dataset({
        features: ['messages'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['messages'],
        num_rows: 2500
    })
})
"""