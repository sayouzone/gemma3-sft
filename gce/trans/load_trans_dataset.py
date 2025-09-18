from datasets import load_dataset

model_name = "gemma-3-12b"
local_csv_file = "datasets/The_Wonderful_Wizard_of_Oz_1.csv"
dataset_name ="philschmid/gretel-synthetic-text-to-sql"
output_dir =f"{model_name}-trans-en-ko"

def sql_prompt(data):
    # System message for the assistant
    system_message = """You are a text to SQL query translator. 
Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

    # User prompt that combines the user query and the schema
    user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, 
generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

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

def translation_prompt(data):
    """데이터셋 샘플을 명령어 형식의 프롬프트로 변환하는 함수"""

    user_prompt = """### Instruction:
Translate the following text from English to Korean.

### Input:
{english_sentence}

### Response:
{korean_sentence}"""

    return {
        "text": user_prompt.format(
            english_sentence=data['english'], 
            korean_sentence=data['korean'])
    }

def translation_genre_prompt(data):
    """데이터셋 샘플을 명령어 형식의 프롬프트로 변환하는 함수"""

    user_prompt = """### Instruction:
Translate the following text from English to Korean as {genre} genre.

### Input:
{english_sentence}

### Response:
{korean_sentence}"""

    return {
        "text": user_prompt.format(
            genre=data['genre'],
            english_sentence=data['english'],
            korean_sentence=data['korean']
        )
    }

def load_dataset_file(file_paths):
    # 단일 파일 로드
    raw_dataset_dict = load_dataset('csv', data_files=file_paths)
    raw_dataset = raw_dataset_dict['train']

    # 여러 파일 한번에 로드 (train/test 분리)
    #data_files = {"train": "train.csv", "test": "test.csv"}
    #dataset = load_dataset('csv', data_files=data_files)

    print(raw_dataset)
    return raw_dataset

# Load dataset from the hub
#dataset = load_dataset(dataset_name, split="train")
#dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
#dataset = dataset.map(sql_prompt, remove_columns=dataset.features,batched=False)
# split dataset into 10,000 training samples and 2,500 test samples
#dataset = dataset.train_test_split(test_size=2500/12500)

# Print formatted user prompt
#print(dataset["train"][345]["messages"][1]["content"])

dataset = load_dataset_file(local_csv_file)
dataset = dataset.map(translation_genre_prompt, remove_columns=dataset.features,batched=False)
