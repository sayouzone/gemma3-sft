#

Ubuntu 24.04 LTS Minimal
x86/64, amd64 noble minimal image
500GB

##

```bash
sudo apt update
sudo apt -y upgrade
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
sudo apt install -y nano screen
```

NVIDIA 드라이버 설치 후 재부팅

```bash
sudo apt update
sudo apt upgrade
sudo ubuntu-drivers autoinstall
sudo reboot

nvidia-smi
```

```bash
Sun Aug 17 23:21:13 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.03              Driver Version: 575.64.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   52C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

```bash
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

문제 해결: reboot Compute Engine

```bash
sudo reboot
```

**Python 3.12 Ubuntu 패키지 설치**

```bash
sudo apt install -y python3.12 python3-pip python3-venv
```

**Python 3.12 수작업 설치**

```bash
wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tar.xz  # 또는 최신 버전 다운로드
tar -xf Python-3.12.0.tar.xz

cd Python-3.12.0
./configure --enable-optimizations
make -j 8  # CPU 코어 수에 맞게 조정
sudo make altinstall

python3 --version
python --version
pip --version
```

```bash
python3 -m venv gemma3-env
source gemma3-env/bin/activate
```

```bash
mkdir gemma3
cd gemma3
```

**전체 파일 복사**

```bash
gcloud storage cp "gs://sayouzone-ai-gemma3/gce-us-central1/*" .

gcloud storage cp * gs://sayouzone-ai-gemma3/gce-us-central1/
```

**일부 파일 복사**

```bash
gcloud storage cp "gs://sayouzone-ai-gemma3/gce-us-central1/*.py" .

gcloud storage cp *.py gs://sayouzone-ai-gemma3/gce-us-central1/
gcloud storage cp requirements.txt gs://sayouzone-ai-gemma3/gce-us-central1/
```

```bash
pip install -r requirements.txt
```

.env 파일 생성

```text
HF_TOKEN=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
```

#### Datasets

Synthetic Text-to-SQL dataset consisting of 105,851 high-quality records across 100 diverse domains, designed for training language models. 

[philschmid/gretel-synthetic-text-to-sql](https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql)

| id (int32) | domain (string) | domain_description (string) | sql_complexity (string) | sql_complexity_description (string) | sql_task_type (string) | sql_task_type_description (string) | sql_prompt (string) | sql_context (string) | sql (string) | sql_explanation (string) |
|------|------|------|------|------|------|------|------|------------|------|------|
| 5,097 | forestry | Comprehensive data on sustainable forest management, timber production, wildlife habitat, and carbon sequestration in forestry. | single join | only one join (specify inner, outer, cross) | analytics and reporting | generating reports, dashboards, and analytical insights | What is the total volume of timber sold by each salesperson, sorted by salesperson? | CREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01'); | SELECT salesperson_id, name, SUM(volume) as total_volume FROM timber_sales JOIN salesperson ON timber_sales.salesperson_id = salesperson.salesperson_id GROUP BY salesperson_id, name ORDER BY total_volume DESC; | Joins timber_sales and salesperson tables, groups sales by salesperson, calculates total volume sold by each salesperson, and orders the results by total volume in descending order. |
| 5,098 | defense industry | Defense contract data, military equipment maintenance, threat intelligence metrics, and veteran employment stats. | aggregation | aggregation functions (COUNT, SUM, AVG, MIN, MAX, etc.), and HAVING clause | analytics and reporting | generating reports, dashboards, and analytical insights | List all the unique equipment types and their corresponding total maintenance frequency from the equipment_maintenance table. | CREATE TABLE equipment_maintenance (equipment_type VARCHAR(255), maintenance_frequency INT); | SELECT equipment_type, SUM(maintenance_frequency) AS total_maintenance_frequency FROM equipment_maintenance GROUP BY equipment_type; | This query groups the equipment_maintenance table by equipment_type and calculates the sum of maintenance_frequency for each group, then returns the equipment_type and the corresponding total_maintenance_frequency. |


**Amazon Multimodal Product dataset**

[philschmid/amazon-product-descriptions-vlm](https://huggingface.co/datasets/philschmid/amazon-product-descriptions-vlm)

| image (image) | Uniq Id (string) | Product Name (string) | Category (string) | Selling Price (string) | Model Number (string) | About Product (string) | Product Specification (string) | Technical Details (string) | Shipping Weight (string) | Variants (string) | Product Url (string) | Is Amazon Seller (string) | description (string) |
|-----|-------|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----------|--------|--------|
| ![1](https://datasets-server.huggingface.co/assets/philschmid/amazon-product-descriptions-vlm/--/f08a021c69c51d6894cfe39206448e7785d6156b/--/default/train/0/image/image.jpg?Expires=1755648845&Signature=QrOz5t2fXjX7p3qGoDDjhw1OY7~pR-otbIUo6HgWPGwqw6zX64K8BPx4mbuinsNALl0UoPZYmf4~gTfr6CRDZ7HUUsQGcTyEA2N6Bbd0a4WJj5uN-3Lyku19HFZRH2PuNk18YeXvcfDU~z31H5qUizn8LAaCuR0glc4UsXvgnvRjeAHR-V94pXikGL6pjQ~aM80iENLePVo~g8~w8AFJuaXvypLn1VR37w7rzos9XHF2aLkML61KL5v2XUBy6rH3NUDOviNb8NRWw8bh5jLQfrpOA~wNMBCcH0r6yorLFUpy91NsuPHWU8yelftypCXo1gW-VfEImASx4k99mni7UA__&Key-Pair-Id=K3EI6M078Z3AC3) | 002e4642d3ead5ecdc9958ce0b3a5a79 | Kurio Glow Smartwatch for Kids with Bluetooth, Apps, Camera & Games, Blue | Toys & Games | Kids' Electronics | Electronic Learning Toys | $31.30 | C17515 | Make sure this fits by entering your model number. | Kurio watch glow is a real Bluetooth Smartwatch built especially for kids, packed with 20+ apps & games! | Get your glow on with new light-up feature that turns games and activities into colorful fun. | Kurio watch glow includes brand-new games with light effects, including the My little dragon virtual pet and color-changing mood sensor. | Play single and two-player games on one watch, Or connect two watches together via Bluetooth, plus motion-sensitive games that get kids moving! | Take fun selfies with the front-facing camera and decorate them with filters, frames and stickers. | Plus, everything you need in a smartwatch – activity tracker, music player, Alarm/stopwatch, calculator, calendar and so much more! | Scratch resistant and splash-proof - suitable for kids ages 4 and up! | ProductDimensions:5x3x12inches|ItemWeight:7.2ounces|ShippingWeight:7.2ounces(Viewshippingratesandpolicies)|ASIN:B07TFD5D55|Itemmodelnumber:C17515|Manufacturerrecommendedage:4yearsandup|Batteries:1LithiumPolymerbatteriesrequired.(included) | Color:Blue show up to 2 reviews by default This sleek, hi-tech Bluetooth Smartwatch is made specifically for kids, and it's packed with apps and games for out-of-the-box fun! Take selfies and videos, play single and two-player games, message friends, listen to music, plus everything you need in a smartwatch– activity tracker, alarm/stopwatch, calculator, calendar and so much more! Plus, parents can add vital information like blood type and allergies to an 'in case of an emergency' (I. C. E. ) app | 7.2 ounces (View shipping rates and policies) | 7.2 ounces | https://www.amazon.com/Kurio-Smartwatch-Bluetooth-Camera-Games/dp/B07TFD5D55|https://www.amazon.com/Kurio-Smartwatch-Bluetooth-Camera-Games/dp/B07TD8JHKW | https://www.amazon.com/Kurio-Smartwatch-Bluetooth-Camera-Games/dp/B07TFD5D55 | Y | Kurio Glow Smartwatch: Fun, Safe & Educational! This kids' smartwatch boasts Bluetooth connectivity, built-in apps & games, and a camera – all in a vibrant blue design. Perfect for learning & play! #kidssmartwatch #kidselectronics #educationaltoys #kurioglow |
| ![2](https://datasets-server.huggingface.co/assets/philschmid/amazon-product-descriptions-vlm/--/f08a021c69c51d6894cfe39206448e7785d6156b/--/default/train/1/image/image.jpg?Expires=1755648845&Signature=MNQ2lTGJxPWdCxj8OPNC5BoCC2sJFiD~1GeIUjwBBxfvwP2pSwE-bPFZHxlYJ5P7gf0eyk-qfvG91OhUMu0sxLWIWBI4kowD0My3DofdTZ4XMYfGNhQR8IouJnSQd8q2Zpa~EvcTYWGRbcuDN8ve-DKhLJ7QlC0Uhl-rONTvn3pizd94DHUGyY5fqlq7QCEVMeX8lGOSKprfSGGMxG3KEvF~wa1WSsFTGzYgGV8LrW1EAFsPNHTZheOkWYFbCCRqwJWGA0AzYjEsl6WoAhxwY1XJ~iuYygrO8ahiSB9M8GdcFogq59IpZ1e-DQEZlcWKxVcuQlhdGpCUmVoGO30ANg__&Key-Pair-Id=K3EI6M078Z3AC3) | 009359198555dde1543d94568183703c | Star Ace Toys Harry Potter & The Prisoner of Azkaban: Harry Potter with Dobby 1: 8 Scale Collectible Action Figure, Multicolor SA8011B | null | $174.99 | SA8011B | Make sure this fits by entering your model number. | From the classic film | Presents Harry as he appeared in the Prisoner Of Azkaban | Comes with the monster book of monsters, hedwig the owl, the Marauder's map and his wand | Includes a figure of the house Elf Dobby | Figure is in 1: 8 scale | ProductDimensions:2.5x1x9inches|ItemWeight:1.43pounds|ShippingWeight:1.43pounds(Viewshippingratesandpolicies)|ASIN:B07KMXGSXF|Itemmodelnumber:SA8011B|Manufacturerrecommendedage:15yearsandup | From Star Ace Toys. Many fans would say that Harry Potter and the Prisoner Of Azkaban is their favorite film of the series. Directed by alfonso cuarón, this film is darker and more adult than the previous two films. Star ace is proud to present Harry Potter as he appeared in this movie in his Hogwarts school robes. He also comes with the monster book of monsters, Hedwig the owl, the Marauder's map and his wand. This special set also includes the house elf Dobby! | 1.43 pounds (View shipping rates and policies) | 1.43 pounds | null | https://www.amazon.com/Star-Ace-Toys-Prisoner-Azkaban/dp/B07KMXGSXF | Y | Relive the magic! Star Ace Toys' 1/8 scale Harry Potter & Dobby collectible figure (SA8011B) from *Prisoner of Azkaban* is here. Highly detailed, this action figure features Harry and his loyal house-elf, Dobby. A must-have for Harry Potter collectors! |
| ![3](https://datasets-server.huggingface.co/assets/philschmid/amazon-product-descriptions-vlm/--/f08a021c69c51d6894cfe39206448e7785d6156b/--/default/train/2/image/image.jpg?Expires=1755648845&Signature=Xc3ESJekJbCDEnv0mPoTBrTYFoX2a4e3Nr3ag6G8tESOCzJPNr~AobVDjdlzHGtKFwgF84InjG-ShO~f~S~bvmHMQXzaRZvHai3mh~EZqpNOmI0TIjmg~i7GNbetuGI66aHHr0SI9VFT2dHi05RT1BqZQtwqdCUt5JxSsbcbS6JQPrzF9pPXWAwQleNzc7iB2PYAv3-soawpmGedlfB3kXYSAkEF~kEk6wzcxrKfAGWT9jfESrUE9yDcbtAg07-LHOGzjHJ0KL8Hm2PvYKNmLDsVdloN-498BBk~JdWzkkMX-zhznjgvMFEK8B212-46vD3umVI5UdtP7BxxNtW71A__&Key-Pair-Id=K3EI6M078Z3AC3) | 00cb3b80482712567c2180767ec28a6a | Barbie Fashionistas Doll Wear Your Heart | Toys & Games | Dolls & Accessories | Dolls | $15.99 | FJF44 | Make sure this fits by entering your model number. | Barbie Fashionistas doll loves this outfit -- the pink Sweatshirt dress has a cool "love" typographic with a sheen touch | Boots and a choker complete the look | Her long hair is right on trend in high pigtails | More variety makes collecting Barbie Fashionistas dolls even more fun | Collect them all (each sold separately, subject to availability) | ProductDimensions:2.1x4.5x12.8inches|ItemWeight:4.2ounces|ShippingWeight:4.2ounces(Viewshippingratesandpolicies)|DomesticShipping:ItemcanbeshippedwithinU.S.|InternationalShipping:ThisitemcanbeshippedtoselectcountriesoutsideoftheU.S.LearnMore|ASIN:B0751ZH2ZT|Itemmodelnumber:FJF44|Manufacturerrecommendedage:36months-7years | Go to your orders and start the return Select the ship method Ship it! | Go to your orders and start the return Select the ship method Ship it! | show up to 2 reviews by default Every Barbie fashionistas doll has her own look from casually cool to boho bold, all fashions are inspired by the latest trends. Collect them all to explore countless styles, fashions, shoes and accessories. The latest line of Barbie fashionistas dolls includes four body types, nine skin tones, 13 eye colors, 13 hairstyles and countless on-trend fashions and accessories. With these additions, girls everywhere will have infinitely more ways to play out their stories and spark their imaginations through Barbie -- because with Barbie, you can be anything! each sold separately, subject to availability. Dolls cannot stand alone. Clothing is designed to mix and match with dolls of the same body Type; select pieces can be shared across the line. Flat shoes fit dolls with articulated ankles or flat feet. Colors and decorations may vary. | 4.2 ounces (View shipping rates and policies) | 4.2 ounces | null | https://www.amazon.com/Barbie-FJF44-Love-Fashion-Doll/dp/B0751ZH2ZT | Y | Express your style with Barbie |


```bash
python3 load_model.py
```

```bash
Traceback (most recent call last):
  File "/home/sjkim/gemma3/load_model.py", line 36, in <module>
    model = model_class.from_pretrained(model_id, **model_kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 547, in from_pretrained
    config, kwargs = AutoConfig.from_pretrained(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py", line 1250, in from_pretrained
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/configuration_utils.py", line 649, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/configuration_utils.py", line 708, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/utils/hub.py", line 321, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/utils/hub.py", line 543, in cached_files
    raise OSError(
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/google/gemma-3-1b-pt.
401 Client Error. (Request ID: Root=1-689eeb63-2e65ce6c208fdc3759534491;fd3df107-1df3-4f14-9baf-a502458eb870)

Cannot access gated repo for url https://huggingface.co/google/gemma-3-1b-pt/resolve/main/config.json.
Access to model google/gemma-3-1b-pt is restricted. You must have access to it and be authenticated to access it. Please log in.
```

```bash
export HF_TOKEN=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
```

```bash
python3 finetune_sql_sfttrainer.py
```

```bash
README.md: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 737/737 [00:00<00:00, 7.47MB/s]
(…)nthetic_text_to_sql_train.snappy.parquet: 100%|█████████████████████████████████████████████████████████████████████████████████████| 32.4M/32.4M [00:00<00:00, 158MB/s]
(…)ynthetic_text_to_sql_test.snappy.parquet: 100%|████████████████████████████████████████████████████████████████████████████████████| 1.90M/1.90M [00:00<00:00, 81.5MB/s]
Generating train split: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 100000/100000 [00:00<00:00, 597090.50 examples/s]
Generating test split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 5851/5851 [00:00<00:00, 434098.19 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 8337.65 examples/s]
SELECT Country, AccessToCleanWater FROM WaterAccess WHERE Continent = 'South America';
model_id google/gemma-3-4b-pt
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 815/815 [00:00<00:00, 8.18MB/s]
model.safetensors.index.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 90.6k/90.6k [00:00<00:00, 6.72MB/s]
model-00002-of-00002.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 3.64G/3.64G [00:36<00:00, 100MB/s]
model-00001-of-00002.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 4.96G/4.96G [00:47<00:00, 105MB/s]
Fetching 2 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:47<00:00, 23.59s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:10<00:00,  5.25s/it]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [00:00<00:00, 935kB/s]
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.16M/1.16M [00:00<00:00, 12.6MB/s]
tokenizer.model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.69M/4.69M [00:00<00:00, 9.63MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33.4M/33.4M [00:00<00:00, 74.4MB/s]
added_tokens.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35.0/35.0 [00:00<00:00, 276kB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 662/662 [00:00<00:00, 4.29MB/s]
```

```bash
    max_seq_length=512,                     # max sequence length for model and packing of the dataset
```

```bash
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
Traceback (most recent call last):
  File "/home/sjkim/gemma3/finetune_sql_sfttrainer.py", line 140, in <module>
    trainer.train()
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/trainer.py", line 2229, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/trainer.py", line 2582, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/trl/trainer/sft_trainer.py", line 904, in training_step
    return super().training_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/trainer.py", line 3845, in training_step
    self.accelerator.backward(loss, **kwargs)
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/accelerate/accelerator.py", line 2730, in backward
    self.scaler.scale(loss).backward(**kwargs)
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/_tensor.py", line 647, in backward
    torch.autograd.backward(
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/autograd/__init__.py", line 354, in backward
    _engine_run_backward(
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/autograd/graph.py", line 829, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.28 GiB is free. Including non-PyTorch memory, this process has 12.28 GiB memory in use. Of the allocated memory 12.06 GiB is allocated by PyTorch, and 93.12 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
  0%|          | 0/3270 [00:02<?, ?it/s]
```

```bash
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7419.86 examples/s]
SELECT SUM(incidents) FROM fire_incidents WHERE city = 'Dallas' AND month = 3 AND year = 2022;
model_id google/gemma-3-1b-pt
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 880/880 [00:00<00:00, 6.10MB/s]
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 2.00G/2.00G [00:04<00:00, 485MB/s]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [00:00<00:00, 1.61MB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1.16M/1.16M [00:00<00:00, 17.2MB/s]
tokenizer.model: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 4.69M/4.69M [00:00<00:00, 11.7MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 33.4M/33.4M [00:00<00:00, 95.3MB/s]
added_tokens.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 35.0/35.0 [00:00<00:00, 267kB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 662/662 [00:00<00:00, 4.19MB/s]
/home/sjkim/gemma3-env/lib/python3.12/site-packages/trl/trainer/sft_trainer.py:453: UserWarning: Padding-free training is enabled, but the attention implementation is not set to 'flash_attention_2'. Padding-free training flattens batches into a single sequence, and 'flash_attention_2' is the only known attention mechanism that reliably supports this. Using other implementations may lead to unexpected behavior. To ensure compatibility, set `attn_implementation='flash_attention_2'` in the model configuration, or verify that your attention mechanism can handle flattened sequences.
  warnings.warn(
/home/sjkim/gemma3-env/lib/python3.12/site-packages/trl/trainer/sft_trainer.py:495: UserWarning: You are using packing, but the attention implementation is not set to 'flash_attention_2' or 'kernels-community/vllm-flash-attn3'. Packing flattens batches into a single sequence, and Flash Attention is the only known attention mechanisms that reliably support this. Using other implementations may lead to cross-contamination between batches. To avoid this, either disable packing by setting `packing=False`, or set `attn_implementation='flash_attention_2'` or `attn_implementation='kernels-community/vllm-flash-attn3'` in the model configuration.
  warnings.warn(
Tokenizing train dataset: 100%|██████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:07<00:00, 1251.06 examples/s]
Packing train dataset: 100%|████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 72879.90 examples/s]
  0%|                                                                                                                                  | 0/3276 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
{'loss': 1.1836, 'grad_norm': 2.295353651046753, 'learning_rate': 0.0002, 'num_tokens': 19319.0, 'mean_token_accuracy': 0.794243934750557, 'epoch': 0.01}
```

```bash
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7946.60 examples/s]
SELECT sales.drug_class FROM sales INNER JOIN drug_info ON sales.drug_class = drug_info.drug_class GROUP BY sales.drug_class HAVING COUNT(sales.id) > 500 WHERE drug_info.drug_category = 'CNS';
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Converting train dataset to ChatML: 100%|██████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 10983.94 examples/s]
Applying chat template to train dataset: 100%|██████████████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 7128.33 examples/s]
Tokenizing train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:05<00:00, 1748.59 examples/s]
Packing train dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:05<00:00, 1877.51 examples/s]
{'loss': 1.6515, 'grad_norm': 2.5932161808013916, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.7348142549395561, 'epoch': 0.01}                                   

...
 21%|██████████████████████████████████                                                                                                                                  | 656/3159 [23:02<1:27:40,  2.10s/it]
...
 43%|██████████████████████████████████████████████████████████████████████▎                                                                                            | 1363/3159 [48:03<1:03:18,  2.12s/it]
...
{'loss': 0.3077, 'grad_norm': 0.8087553381919861, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.9060454100370408, 'epoch': 2.99}
{'train_runtime': 6704.5997, 'train_samples_per_second': 1.885, 'train_steps_per_second': 0.471, 'train_loss': 0.39185931562584914, 'mean_token_accuracy': 0.9053503606054518, 'epoch': 3.0}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3159/3159 [1:51:44<00:00,  2.12s/it
```

```bash
python3 test_sql_inference.py 
Using device: gpu
Using dtype: torch.bfloat16
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Device set to use cuda:0
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7920.86 examples/s]
INSERT INTO New_Mine_Productivity (Mine_Name, Productivity, Year) VALUES ('Amethyst Ascent', 5.9, 2019);
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Context:
 CREATE TABLE threats (threat_type VARCHAR(255), organization VARCHAR(255), threat_date DATE); INSERT INTO threats (threat_type, organization, threat_date) VALUES ('Phishing', 'Org789', '2022-01-01'), ('Malware', 'Org789', '2022-01-05'), ('Ransomware', 'Org789', '2022-01-10'), ('Phishing', 'Org789', '2022-02-01'), ('Phishing', 'Org789', '2022-02-15'), ('Malware', 'Org789', '2022-03-01'), ('Phishing', 'Org789', '2022-03-15'), ('Ransomware', 'Org789', '2022-04-01'), ('Phishing', 'Org789', '2022-04-15'), ('Malware', 'Org789', '2022-05-01'), ('Phishing', 'Org789', '2022-05-15');
Query:
 Identify the top 3 most frequent types of threats and their frequency for the organization 'Org789' for the current year?
Original Answer:
SELECT threat_type, COUNT(threat_type) as frequency FROM threats WHERE organization = 'Org789' AND threat_date >= DATEADD(year, -1, GETDATE()) GROUP BY threat_type ORDER BY frequency DESC LIMIT 3;
Generated Answer:
Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
CREATE TABLE marine_protected_areas (name VARCHAR(255), ocean VARCHAR(255), depth FLOAT, region VARCHAR(255)); INSERT INTO marine_protected_areas (name, ocean, depth, region) VALUES ('Galapagos Marine Reserve', 'Pacific Ocean', 1000.0, 'Pacific Ocean');
</SCHEMA>

<USER_QUERY>
What is the average depth of marine protected areas in the Pacific Ocean, grouped by region?
</USER_QUERY>
```

gemma-3-4b-pt

```bash
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7642.47 examples/s]
SELECT City, MAX(Fare) as MaxFare FROM Fares GROUP BY City;
model_id google/gemma-3-4b-pt

{'loss': 1.0427, 'grad_norm': 1.9847701787948608, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.920499025285244, 'epoch': 2.86}                                                                                      
 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████        | 3006/3150 [2:38:16<07:35,  3.16s/it]

{'train_runtime': 9991.0722, 'train_samples_per_second': 1.261, 'train_steps_per_second': 0.315, 'train_loss': 1.3754941192505852, 'epoch': 3.0}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3150/3150 [2:46:31<00:00,  3.17s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:26<00:00, 13.07s/it]
```

```bash
Using device: gpu
Using dtype: torch.bfloat16
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:29<00:00, 14.58s/it]
Device set to use cuda:0
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:06<00:00, 1961.52 examples/s]
UPDATE hr.employees SET hire_date = '2019-06-01' WHERE id = 1;
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Context:
 CREATE TABLE co_ownership (id INT, neighborhood VARCHAR(50), co_owned BOOLEAN); INSERT INTO co_ownership (id, neighborhood, co_owned) VALUES (1, 'Westwood', TRUE), (2, 'Beverly Hills', FALSE), (3, 'Venice', TRUE);
Query:
 How many properties are co-owned in each neighborhood?
Original Answer:
SELECT neighborhood, COUNT(*) OVER (PARTITION BY co_owned) AS co_owned_count FROM co_ownership;
Generated Answer:
user
Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
CREATE TABLE Employees (EmployeeID INT, Department VARCHAR(50), Salary FLOAT); INSERT INTO Employees (EmployeeID, Department, Salary) VALUES (1, 'IT', 75000.00), (2, 'IT', 80000.00), (3, 'IT', 70000.00), (4, 'IT', 90000.00), (5, 'IT', 65000.00);
</SCHEMA>

<USER_QUERY>
What is the average salary for employees in the IT department?
</USER_QUERY>
```


```bash
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7642.47 examples/s]
SELECT City, MAX(Fare) as MaxFare FROM Fares GROUP BY City;
model_id google/gemma-3-12b-pt
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 876/876 [00:00<00:00, 4.67MB/s]
model.safetensors.index.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 109k/109k [00:00<00:00, 4.56MB/s]
model-00005-of-00005.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.60G/4.60G [02:07<00:00, 36.1MB/s]
model-00001-of-00005.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.98G/4.98G [02:19<00:00, 35.6MB/s]
model-00003-of-00005.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.93G/4.93G [02:20<00:00, 35.1MB/s]
model-00002-of-00005.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.93G/4.93G [02:28<00:00, 33.3MB/s]
model-00004-of-00005.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.93G/4.93G [02:28<00:00, 33.3MB/s]
Fetching 5 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [02:28<00:00, 29.67s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:17<00:00, 15.46s/it]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [00:00<00:00, 1.25MB/s]
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.16M/1.16M [00:00<00:00, 22.0MB/s]
tokenizer.model: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.69M/4.69M [00:00<00:00, 10.1MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33.4M/33.4M [00:00<00:00, 72.5MB/s]
added_tokens.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35.0/35.0 [00:00<00:00, 313kB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 662/662 [00:00<00:00, 5.75MB/s]
Traceback (most recent call last):
  File "/home/sjkim/gemma3/finetune_sql_sfttrainer.py", line 119, in <module>
    trainer = SFTTrainer(
              ^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/trl/trainer/sft_trainer.py", line 183, in __init__
    model = self._prepare_peft_model(model, peft_config, args)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/trl/trainer/sft_trainer.py", line 320, in _prepare_peft_model
    model = get_peft_model(model, peft_config)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/peft/mapping.py", line 222, in get_peft_model
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/peft/peft_model.py", line 1684, in __init__
    super().__init__(model, peft_config, adapter_name, **kwargs)
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/peft/peft_model.py", line 176, in __init__
    self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/peft/tuners/lora/model.py", line 141, in __init__
    super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/peft/tuners/tuners_utils.py", line 184, in __init__
    self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/peft/tuners/tuners_utils.py", line 483, in inject_adapter
    new_module = ModulesToSaveWrapper(target, adapter_name)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/peft/utils/other.py", line 212, in __init__
    self.update(adapter_name)
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/peft/utils/other.py", line 278, in update
    self.modules_to_save.update(torch.nn.ModuleDict({adapter_name: copy.deepcopy(self.original_module)}))
                                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/copy.py", line 162, in deepcopy
    y = _reconstruct(x, memo, *rv)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/copy.py", line 259, in _reconstruct
    state = deepcopy(state, memo)
            ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/copy.py", line 136, in deepcopy
    y = copier(x, memo)
        ^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/copy.py", line 221, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
                             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/copy.py", line 136, in deepcopy
    y = copier(x, memo)
        ^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/copy.py", line 221, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
                             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/copy.py", line 143, in deepcopy
    y = copier(memo)
        ^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/nn/parameter.py", line 68, in __deepcopy__
    self.data.clone(memory_format=torch.preserve_format), self.requires_grad
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.75 GiB. GPU 0 has a total capacity of 22.05 GiB of which 3.62 GiB is free. Including non-PyTorch memory, this process has 18.42 GiB memory in use. Of the allocated memory 13.05 GiB is allocated by PyTorch, and 5.16 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

```bash
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = "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"
```

```bash
Map: 100%|███████████████████████████████████████████████████████████████████████| 12500/12500 [00:02<00:00, 5899.16 examples/s]
INSERT INTO inspections (restaurant_name, grade, inspection_date) VALUES ('ABC Restaurant', 'B', '2023-02-15');
model_id google/gemma-3-12b-pt
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:31<00:00, 18.23s/it]
Converting train dataset to ChatML: 100%|███████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 10844.00 examples/s]
Applying chat template to train dataset: 100%|███████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 6995.95 examples/s]
Tokenizing train dataset: 100%|██████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:05<00:00, 1735.60 examples/s]
Packing train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:05<00:00, 1867.54 examples/s]
Traceback (most recent call last):
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/huggingface_hub/utils/_http.py", line 409, in hf_raise_for_status
    response.raise_for_status()
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/requests/models.py", line 1026, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/api/repos/create

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/sjkim/gemma3/finetune_sql_sfttrainer.py", line 122, in <module>
    trainer = SFTTrainer(
              ^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/trl/trainer/sft_trainer.py", line 232, in __init__
    super().__init__(
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/trainer.py", line 699, in __init__
    self.init_hf_repo()
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/trainer.py", line 4725, in init_hf_repo
    repo_url = create_repo(repo_name, token=token, private=self.args.hub_private_repo, exist_ok=True)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/huggingface_hub/hf_api.py", line 3755, in create_repo
    hf_raise_for_status(r)
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/huggingface_hub/utils/_http.py", line 482, in hf_raise_for_status
    raise _format(HfHubHTTPError, str(e), response) from e
huggingface_hub.errors.HfHubHTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/api/repos/create (Request ID: Root=1-689ff10b-34e22bc516f5bf49705b7945;086a3aa6-2115-4baa-878d-8bb18dfa1d7b)

Invalid username or password.
```

```bash
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7946.60 examples/s]
SELECT sales.drug_class FROM sales INNER JOIN drug_info ON sales.drug_class = drug_info.drug_class GROUP BY sales.drug_class HAVING COUNT(sales.id) > 500 WHERE drug_info.drug_category = 'CNS';
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
model_id google/gemma-3-1b-pt
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:31<00:00, 18.23s/it]
Converting train dataset to ChatML: 100%|███████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 10844.00 examples/s]
Applying chat template to train dataset: 100%|███████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 6995.95 examples/s]
Tokenizing train dataset: 100%|██████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:05<00:00, 1735.60 examples/s]
Packing train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:05<00:00, 1867.54 examples/s]

{'loss': 0.2865, 'grad_norm': 0.808881938457489, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.9148238748311996, 'epoch': 2.83}                                                                          
 94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉         | 2983/3159 [1:47:11<06:14,  2.13s/it

{'loss': 0.3007, 'grad_norm': 0.7644242644309998, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.9109589010477066, 'epoch': 2.99}
{'train_runtime': 6833.4027, 'train_samples_per_second': 1.848, 'train_steps_per_second': 0.462, 'train_loss': 0.39142834170402774, 'mean_token_accuracy': 0.9111315730740043, 'epoch': 3.0}                 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3159/3159 [1:53:53<00:00,  2.16s/it]
```

```bash
model_id google/gemma-3-1b-pt
Using device: gpu
Using dtype: torch.bfloat16
Device set to use cuda:0
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7665.18 examples/s]
SELECT gm.name, SUM(grant_amount) as total_granted FROM grant_makers gm INNER JOIN grants g ON gm.id = g.grant_maker_id WHERE g.grant_date <= CURRENT_DATE AND g.grant_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR) GROUP BY gm.id;
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Context:
 CREATE TABLE machine_usage (machine TEXT, usage INTEGER, start_time TIMESTAMP, end_time TIMESTAMP);
Query:
 Show the number of working hours for each agricultural machine in the past week.
Original Answer:
SELECT machine, SUM(DATEDIFF(minute, start_time, end_time)) as working_hours FROM machine_usage WHERE end_time BETWEEN DATEADD(day, -7, CURRENT_TIMESTAMP) AND CURRENT_TIMESTAMP GROUP BY machine;
Generated Answer:
SELECT machine, SUM(DATEDIFF(hour, start_time, end_time)) as total_hours FROM machine_usage WHERE start_time BETWEEN DATEADD(day, -7, CURRENT_TIMESTAMP) AND CURRENT_TIMESTAMP GROUP BY machine
```

```bash
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7888.20 examples/s]
SELECT MIN(salary) FROM manufacturing_union;
model_id google/gemma-3-4b-pt
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:32<00:00, 16.09s/it]
Converting train dataset to ChatML: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 9239.61 examples/s]
Applying chat template to train dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 7003.75 examples/s]
Tokenizing train dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:05<00:00, 1759.58 examples/s]
Packing train dataset: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:05<00:00, 1875.67 examples/s]
  0%|                                                                                                                                                                               | 0/3150 [00:00<?, ?it/s]
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  0%|▍                                                                                                                                                                    | 9/3150 [00:29<2:46:56,  3.19s/it

{'train_runtime': 10199.5493, 'train_samples_per_second': 1.235, 'train_steps_per_second': 0.309, 'train_loss': 1.3731104118104964, 'epoch': 3.0}                                                            
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3150/3150 [2:49:59<00:00,  3.24s/it]
No files have been modified since last commit. Skipping to prevent empty commit.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.15s/it]
```

gemma-3-4b-text-to-sql

```bash
model_id google/gemma-3-4b-pt
Using device: gpu
Using dtype: torch.bfloat16
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:29<00:00, 14.70s/it]
Device set to use cuda:0
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:02<00:00, 5819.67 examples/s]
SELECT country, SUM(funding) FROM climate_mitigation_projects WHERE year = 2020 GROUP BY country;
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Context:
 CREATE TABLE Aircraft (id INT, model VARCHAR(255), manufacturer VARCHAR(255), year_manufactured INT, total_flight_hours INT); INSERT INTO Aircraft (id, model, manufacturer, year_manufactured, total_flight_hours) VALUES (1, 'B747', 'Boeing', 1990, 50000); INSERT INTO Aircraft (id, model, manufacturer, year_manufactured, total_flight_hours) VALUES (2, 'A320', 'Airbus', 2005, 30000); CREATE TABLE Engine (id INT, aircraft_id INT, engine_type VARCHAR(255), hours_since_last_service INT); INSERT INTO Engine (id, aircraft_id, engine_type, hours_since_last_service) VALUES (1, 1, 'CF6-80C2B1', 500); INSERT INTO Engine (id, aircraft_id, engine_type, hours_since_last_service) VALUES (2, 2, 'CFM56-5B', 1000); INSERT INTO Engine (id, aircraft_id, engine_type, hours_since_last_service) VALUES (3, 1, 'CF6-80C2B1', 700);
Query:
 What is the total number of engines for each aircraft?
Original Answer:
SELECT a.model, COUNT(e.id) FROM Aircraft a JOIN Engine e ON a.id = e.aircraft_id GROUP BY a.model;
Generated Answer:
SELECT a.model, COUNT(e.id) FROM Aircraft a LEFT JOIN Engine e ON a.id = e.aircraft_id GROUP BY a.model;
```

gemma-3-4b_merged_model

```bash
model_id google/gemma-3-4b-pt
Using device: gpu
Using dtype: torch.bfloat16
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:07<00:00,  6.76s/it]
Device set to use cuda:0
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7602.23 examples/s]
SELECT COUNT(*) FROM Safety_Protocols WHERE department = 'Metal Fabrication' AND protocol_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Context:
 CREATE TABLE emergencies (type VARCHAR(255), response_time INT); INSERT INTO emergencies (type, response_time) VALUES ('Fire', 5), ('Medical', 8);
Query:
 What is the difference in response time between fire and medical emergencies?
Original Answer:
SELECT type, LEAD(response_time) OVER (ORDER BY response_time) - response_time AS difference FROM emergencies;
Generated Answer:
SELECT type, LEAD(response_time) OVER (ORDER BY response_time) - response_time AS difference FROM emergencies;
```

**gemma3-n1s8-t4-test (n1-highmem-8 (vCPU 8개, 메모리 52GB) + NVIDIA T4)**

google/gemma-3-1b-pt

```bash
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7484.26 examples/s]
SELECT sensor_name, measurement FROM PrecisionFarming.IoT_Sensors WHERE measurement = 'moisture' OR measurement = 'temperature';
model_id google/gemma-3-1b-pt
Tokenizing train dataset: 100%|██████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:08<00:00, 1240.67 examples/s]
Truncating train dataset: 100%|████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 166702.60 examples/s]
  0%|                                                                                                                                  | 0/7500 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
{'loss': 1.4642, 'grad_norm': 2.6335740089416504, 'learning_rate': 0.0002, 'num_tokens': 9140.0, 'mean_token_accuracy': 0.7516661927103996, 'epoch': 0.0}       
{'loss': 0.7514, 'grad_norm': 2.020350694656372, 'learning_rate': 0.0002, 'num_tokens': 17263.0, 'mean_token_accuracy': 0.8441643685102462, 'epoch': 0.01}      
{'loss': 0.6605, 'grad_norm': 2.2366912364959717, 'learning_rate': 0.0002, 'num_tokens': 25990.0, 'mean_token_accuracy': 0.8503234028816223, 'epoch': 0.01}     
{'loss': 0.5738, 'grad_norm': 1.7300941944122314, 'learning_rate': 0.0002, 'num_tokens': 34263.0, 'mean_token_accuracy': 0.8595587998628617, 'epoch': 0.02}     
  1%|▋                                                                                                                      | 41/7500 [01:49<5:28:27,  2.64s/it]

{'loss': 0.2766, 'grad_norm': 0.9266780018806458, 'learning_rate': 0.0002, 'num_tokens': 6417433.0, 'mean_token_accuracy': 0.9154374480247498, 'epoch': 3.0}    
{'loss': 0.2634, 'grad_norm': 1.3448530435562134, 'learning_rate': 0.0002, 'num_tokens': 6425859.0, 'mean_token_accuracy': 0.9230536803603172, 'epoch': 3.0}    
{'train_runtime': 19694.5249, 'train_samples_per_second': 1.523, 'train_steps_per_second': 0.381, 'train_loss': 0.3633397448539734, 'epoch': 3.0}               
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7500/7500 [5:28:14<00:00,  2.63s/it]
No files have been modified since last commit. Skipping to prevent empty commit.
```

test_sql_inference.py

```bash
model_id google/gemma-3-1b-pt
Using device: cpu
Using dtype: torch.float32
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  1.65it/s]
Device set to use cuda:0
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7430.72 examples/s]
SELECT SUM(num_attendees) FROM Events WHERE event_location = 'New York' AND event_type <> 'Workshop';
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Context:
 CREATE TABLE site_k_artifacts (id INT PRIMARY KEY, site_id INT, artifact_type VARCHAR(50), quantity INT); INSERT INTO site_k_artifacts (id, site_id, artifact_type, quantity) VALUES (1, 8, 'Stone tool', 15), (2, 8, 'Pottery shard', 20), (3, 8, 'Copper coin', 5), (4, 8, 'Bronze coin', 10);
Query:
 Delete all records with 'Copper coin' artifact_type from the 'site_k_artifacts' table.
Original Answer:
DELETE FROM site_k_artifacts WHERE artifact_type = 'Copper coin';
Generated Answer:
SELECT site_id, artifact_type, quantity FROM site_k_artifacts WHERE artifact_type = 'Copper coin';
```

```bash
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:01<00:00, 7422.83 examples/s]
SELECT (COUNT(*) FILTER (WHERE autonomous_driving = TRUE)) * 100.0 / COUNT(*) FROM ResearchPapers WHERE publication_year = 2021;
model_id google/gemma-3-4b-pt
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:10<00:00,  5.39s/it]
Tokenizing train dataset: 100%|██████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:08<00:00, 1243.02 examples/s]
Truncating train dataset: 100%|████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 182496.73 examples/s]
  0%|                                                                                                                                  | 0/7500 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
Traceback (most recent call last):
  File "/home/sjkim/gemma3/finetune_sql_sfttrainer.py", line 141, in <module>
    trainer.train()
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/trainer.py", line 2229, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/trainer.py", line 2582, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/trl/trainer/sft_trainer.py", line 904, in training_step
    return super().training_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/trainer.py", line 3845, in training_step
    self.accelerator.backward(loss, **kwargs)
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/accelerate/accelerator.py", line 2730, in backward
    self.scaler.scale(loss).backward(**kwargs)
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/_tensor.py", line 647, in backward
    torch.autograd.backward(
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/autograd/__init__.py", line 354, in backward
    _engine_run_backward(
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/autograd/graph.py", line 829, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.27 GiB is free. Including non-PyTorch memory, this process has 12.29 GiB memory in use. Of the allocated memory 12.05 GiB is allocated by PyTorch, and 104.42 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
  0%|          | 0/7500 [00:02<?, ?it/s]
```

finetune_vision_sfttrainer.py

```bash
README.md: 1.22kB [00:00, 6.28MB/s]
train-00000-of-00001.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 47.6M/47.6M [00:00<00:00, 193MB/s]
Generating train split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1345/1345 [00:00<00:00, 9662.03 examples/s]
[{'role': 'system', 'content': [{'type': 'text', 'text': 'You are an expert product description writer for Amazon.'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.\nOnly return description. The description should be SEO optimized and for a better mobile search experience.\n\n<PRODUCT>\nMasterPieces Tribal Spirit Jigsaw Puzzle, The Chiefs, Featuring American Indian Tribe Traditions & Ceremonies, 1000 Pieces\n</PRODUCT>\n\n<CATEGORY>\nToys & Games | Puzzles | Jigsaw Puzzles\n</CATEGORY>\n'}, {'type': 'image', 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x70A6E85E5340>}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Challenge yourself with this 1000-piece MasterPieces Tribal Spirit jigsaw puzzle!  Depicting the rich traditions and ceremonies of American Indian tribes, "The Chiefs" offers a stunning, culturally significant image perfect for puzzle enthusiasts.  High-quality pieces guarantee a satisfying solve.'}]}]
model_id google/gemma-3-4b-pt
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:32<00:00, 16.01s/it]
processor_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 70.0/70.0 [00:00<00:00, 697kB/s]
chat_template.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.61k/1.61k [00:00<00:00, 17.3MB/s]
preprocessor_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 570/570 [00:00<00:00, 5.22MB/s]
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  1%|▊                                                                                                                                              | 2/337 [00:42<1:58:01, 21.14s/it]
{'loss': 9.9039, 'grad_norm': 29.41756248474121, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.5849237442016602, 'epoch': 0.01}                                                   
{'loss': 5.679, 'grad_norm': 17.978452682495117, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.7487379521131515, 'epoch': 0.03}                                                   
{'loss': 5.1366, 'grad_norm': 18.68014144897461, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.7553565979003907, 'epoch': 0.04}

{'loss': 3.115, 'grad_norm': 8.573728561401367, 'learning_rate': 0.0002, 'mean_token_accuracy': 0.8290552139282227, 'epoch': 1.0}                                                     
{'train_runtime': 6918.3363, 'train_samples_per_second': 0.194, 'train_steps_per_second': 0.049, 'train_loss': 3.510115796097662, 'mean_token_accuracy': 0.84203120470047, 'epoch': 1.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 337/337 [1:55:18<00:00, 20.53s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.00s/it]
```

test_vision_inference.py

```bash
[{'role': 'system', 'content': [{'type': 'text', 'text': 'You are an expert product description writer for Amazon.'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.\nOnly return description. The description should be SEO optimized and for a better mobile search experience.\n\n<PRODUCT>\nMasterPieces Tribal Spirit Jigsaw Puzzle, The Chiefs, Featuring American Indian Tribe Traditions & Ceremonies, 1000 Pieces\n</PRODUCT>\n\n<CATEGORY>\nToys & Games | Puzzles | Jigsaw Puzzles\n</CATEGORY>\n'}, {'type': 'image', 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7BFEC8249D60>}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Challenge yourself with this 1000-piece MasterPieces Tribal Spirit jigsaw puzzle!  Depicting the rich traditions and ceremonies of American Indian tribes, "The Chiefs" offers a stunning, culturally significant image perfect for puzzle enthusiasts.  High-quality pieces guarantee a satisfying solve.'}]}]
model_id google/gemma-3-4b-pt
Using device: gpu
Using dtype: torch.bfloat16
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.43s/it]
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.

Bring the Marvel universe home with the Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held Iron Man Actionfigures! This 30.5cm giant Iron Man figure is a must-have for any Marvel fan.  Perfect for imaginative play and collecting action figures.  Shop now!
```

```bash
Loading base model and tokenizer for 'google/gemma-3-4b-it'...
Loading checkpoint shards:   0%|                                                                                                                                                                     | 0/2 [00:00<?, ?it/s]
Loading checkpoint shards:  50%|██████████████████████████████████████████████████████████████████████████████▌                     Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:34<00:00, 17.23s/it]
Model and tokenizer loaded successfully.
Loading and preparing dataset 'lemon-mint/korean_parallel_sentences_v1.1'...
Dataset formatted.
Sample formatted prompt:
### Instruction:
Translate the following text from English to Korean.

### Input:
The uterus is located below the belly button, and its size and shape vary from woman to woman. The uterus is usually pear-shaped and about 7~8 cm in length.

### Response:
자궁은 배꼽 아래에 위치하며, 크기와 모양은 여성마다 다릅니다. 자궁은 일반적으로 배 모양이며, 길이는 약 7~8cm입니다.
Configuring LoRA...
trainable params: 16,394,240 || all params: 4,316,473,712 || trainable%: 0.3798
Starting training...
{'loss': 9.3296, 'grad_norm': 6.3900675773620605, 'learning_rate': 0.0001998765672708374, 'mean_token_accuracy': 0.7619765393435955, 'epoch': 0.0}
  1%|██▏                                                                                                                                                                    | 406/30786 [23:43<29:27:11,  3.49s/it]

{'train_runtime': 108987.221, 'train_samples_per_second': 4.519, 'train_steps_per_second': 0.282, 'train_loss': 5.2918518922996975, 'mean_token_accuracy': 0.8448703629629952, 'epoch': 1.0}                       
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30786/30786 [30:16:27<00:00,  3.54s/it]
Training completed.
Saving final adapter and merging...
Adapter saved to 'gemma-3-4b-en-ko-trans_final_adapter'
Merged model saved to 'gemma-3-4b-en-ko-trans_merged_model'

--- Inference Test with Merged Model ---
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.50s/it]
/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/_inductor/compile_fx.py:282: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
W0823 13:44:00.144000 1256 torch/_inductor/utils.py:1436] [0/0] Not enough SMs to use max_autotune_gemm mode
English Input:
The ability to fine-tune powerful language models on consumer hardware is a significant breakthrough for the AI community.

Korean Translation:
['### Instruction:\nTranslate the following text from English to Korean.\n\n### Input:\nThe ability to fine-tune powerful language models on consumer hardware is a significant breakthrough for the AI community.\n\n', '\n소비자 하드웨어에서 강력한 언어 모델을 미세 조정할 수 있는 능력은 AI 커뮤니티에 큰 이점입니다. 이를 통해 더 많은 사람들이 AI 기술에 접근할 수 있게 되고, AI가 우리 삶의 다양한 측면에 통합될 수 있습니다.\n\n', '\n소비자 하드웨어에서 강력한 언어 모델을 미세 조정할 수 있는 능력은 AI 커뮤니티에 큰 이점이 됩니다. 이를 통해 더 많은 사람들이 AI 기술에 접근할 수 있게 되고, AI가 우리 삶의 다양한 측면에 통합될 수 있습니 다.\n\n', '\n소비자 하드웨어에서 강력한 언어 모델을 미세 조정할 수 있는 능력은 AI 커뮤니티에 큰 이점이 됩니다. 이를 통해 더 많은 사람들이 AI 기술에 접근할 수 있게 되고, AI가 우리 삶의 다양한 측면에 통합될 수 있 습니다.\n\n', '\n소비자 하드웨어에서 강력한 언어 모델을 미세 조정할 수 있는 능력은 AI 커뮤니티에 큰 이점이 됩니다. 이를 통해 더 많은 사람들이 AI 기술에 접근할 수 있게 되고, AI가 우리 삶의 다양한 측면에 통합될 수 있습니다.\n\n', '\n소비자 하드웨어에서 강력한 언어 모델을 미세 조정할 수 있는 능력은 AI 커뮤니티에 큰 이점이 됩니다. 이를 통해 더 많은 사람들이 AI 기술에 접근할 수 있게 되고, AI가 우리 삶의 다양한 측면에 통합될 수 있습니다.\n\n', '\n소비자 하드웨어에서 강력한 언어 모델을 미세 조정할 수 있는 능력은 AI 커뮤니티에 큰 이점이 됩니다. 이를 통해 더 많은 사람들이 AI 기술에 접근할 수 있게 되고,']
```

**Nvidia V100 Translation**
L4보다 번역 파인튜닝에서 성능이 떨어짐<br>
T4보다 약간 성능이 우수

```bash
Loading base model and tokenizer for 'google/gemma-3-4b-it'...
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 855/855 [00:00<00:00, 5.52MB/s]
model.safetensors.index.json: 100%|██████████████████████████████████████████████████████████████████████████| 90.6k/90.6k [00:00<00:00, 6.83MB/s]
model-00002-of-00002.safetensors: 100%|██████████████████████████████████████████████████████████████████████| 3.64G/3.64G [00:42<00:00, 85.0MB/s]
model-00001-of-00002.safetensors: 100%|███████████████████████████████████████████████████████████████████████| 4.96G/4.96G [00:47<00:00, 106MB/s]
Fetching 2 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:47<00:00, 23.60s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.63s/it]
generation_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████| 215/215 [00:00<00:00, 1.89MB/s]
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████| 1.16M/1.16M [00:00<00:00, 12.0MB/s]
tokenizer.model: 100%|███████████████████████████████████████████████████████████████████████████████████████| 4.69M/4.69M [00:00<00:00, 10.3MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 33.4M/33.4M [00:00<00:00, 129MB/s]
added_tokens.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 35.0/35.0 [00:00<00:00, 252kB/s]
special_tokens_map.json: 100%|███████████████████████████████████████████████████████████████████████████████████| 662/662 [00:00<00:00, 4.27MB/s]
Model and tokenizer loaded successfully.
Loading and preparing dataset 'lemon-mint/korean_parallel_sentences_v1.1'...
README.md: 2.48kB [00:00, 11.4MB/s]
data/train-00000-of-00001.parquet: 100%|███████████████████████████████████████████████████████████████████████| 134M/134M [00:01<00:00, 67.1MB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████████████| 492564/492564 [00:00<00:00, 833880.18 examples/s]
Map (num_proc=4): 100%|█████████████████████████████████████████████████████████████████████████| 492564/492564 [00:07<00:00, 69664.37 examples/s]
Dataset formatted.
Sample formatted prompt:
### Instruction:
Translate the following text from English to Korean.

### Input:
The uterus is located below the belly button, and its size and shape vary from woman to woman. The uterus is usually pear-shaped and about 7~8 cm in length.

### Response:
자궁은 배꼽 아래에 위치하며, 크기와 모양은 여성마다 다릅니다. 자궁은 일반적으로 배 모양이며, 길이는 약 7~8cm입니다.
Configuring LoRA...
trainable params: 16,394,240 || all params: 4,316,473,712 || trainable%: 0.3798
Adding EOS to train dataset: 100%|██████████████████████████████████████████████████████████████| 492564/492564 [00:16<00:00, 30717.58 examples/s]
Tokenizing train dataset: 100%|██████████████████████████████████████████████████████████████████| 492564/492564 [03:16<00:00, 2507.39 examples/s]
Truncating train dataset: 100%|████████████████████████████████████████████████████████████████| 492564/492564 [00:02<00:00, 228804.26 examples/s]
Starting training...
  0%|                                                                                                       | 2/30786 [00:36<155:37:45, 18.20s/it]
  0%|                                                                                                       | 6/30786 [01:51<161:51:43, 18.93s/it]

```

Nvidia T4 Translation

```bash
Loading base model and tokenizer for 'google/gemma-3-4b-it'...
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 855/855 [00:00<00:00, 6.84MB/s]
model.safetensors.index.json: 100%|██████████████████████████████████████████████████████████████████████████████| 90.6k/90.6k [00:00<00:00, 4.11MB/s]
model-00001-of-00002.safetensors: 100%|███████████████████████████████████████████████████████████████████████████| 4.96G/4.96G [00:41<00:00, 120MB/s]
model-00002-of-00002.safetensors: 100%|██████████████████████████████████████████████████████████████████████████| 3.64G/3.64G [00:42<00:00, 86.1MB/s]
Fetching 2 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:42<00:00, 21.25s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.62s/it]
generation_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 215/215 [00:00<00:00, 1.48MB/s]
Model and tokenizer loaded successfully.
Loading and preparing dataset 'lemon-mint/korean_parallel_sentences_v1.1'...
README.md: 2.48kB [00:00, 4.73MB/s]
data/train-00000-of-00001.parquet: 100%|███████████████████████████████████████████████████████████████████████████| 134M/134M [00:01<00:00, 70.6MB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████████████████| 492564/492564 [00:00<00:00, 798089.48 examples/s]
Map (num_proc=4): 100%|█████████████████████████████████████████████████████████████████████████████| 492564/492564 [00:06<00:00, 71559.83 examples/s]
Dataset formatted.
Sample formatted prompt:
### Instruction:
Translate the following text from English to Korean.

### Input:
The uterus is located below the belly button, and its size and shape vary from woman to woman. The uterus is usually pear-shaped and about 7~8 cm in length.

### Response:
자궁은 배꼽 아래에 위치하며, 크기와 모양은 여성마다 다릅니다. 자궁은 일반적으로 배 모양이며, 길이는 약 7~8cm입니다.
Configuring LoRA...
trainable params: 16,394,240 || all params: 4,316,473,712 || trainable%: 0.3798
Adding EOS to train dataset: 100%|██████████████████████████████████████████████████████████████████| 492564/492564 [00:15<00:00, 31398.57 examples/s]
Tokenizing train dataset: 100%|██████████████████████████████████████████████████████████████████████| 492564/492564 [03:12<00:00, 2556.60 examples/s]
Truncating train dataset: 100%|████████████████████████████████████████████████████████████████████| 492564/492564 [00:02<00:00, 220913.62 examples/s]
Starting training...
  0%|                                                                                                           | 3/30786 [01:00<171:13:34, 20.02s/it]
  0%|                                                                                                           | 4/30786 [01:20<171:22:58, 20.04s/it]
```