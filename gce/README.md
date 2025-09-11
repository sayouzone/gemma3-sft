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
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

#### Gemma 3 Models

| 크기 | 파라미터 수 | 컨텍스트 창 | 다중 모드 | 16-bit 메모리 |
|------|-----|----------|------|-------|
| 270M | 10억 | 32K 토큰 | 텍스트 | 400MB |
| 1B | 10억 | 32K 토큰 | 텍스트 | 1.5GB |
| 4B | 40억 | 128K 토큰 | 텍스트, 이미지 | 6.4 GB |
| 12B | 120억 | 128K 토큰 | 텍스트, 이미지 | 20 GB |
| 27B | 270억 | 128K 토큰 | 텍스트, 이미지 | 46.4 GB |

#### Datasets (Hugging Face)

| 데이터셋 이름 | 크기 (쌍의 수) | 도메인 | 비고/품질 |
|------------------------------|-------------|------|---------|
| nayohan/aihub-en-ko-translation-1.2m | 1,190,000 | 뉴스, 대화, 위키 | AI Hub에서 제공하는 대규모 고품질 데이터셋. 다양한 도메인을 포함. |
| bongsoo/news_talk_en_ko | 1,300,000 | 뉴스, 구어체 | 뉴스 기사와 대화체 텍스트를 포함하여 공식 및 비공식 번역에 유용. |
| lemon-mint/korean_english_parallel_wiki_augmented_v1 | 503,000 | 위키피디아 | 위키피디아 문서 기반으로, 정보성 텍스트 번역에 적합. |
| Moo/korean-parallel-corpora | 99,000 | 다양함 | 크기는 작지만, 빠른 프로토타이핑이나 특정 스타일 학습에 사용될 수 있음. |


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

