#

Apple Metal Performance Shaders (MPS)
QLoRA(Quantized Low-Rank Adaptation)

출력 디렉토리에는 단 몇 MB 크기의 경량 어댑터 가중치(adapter_model.safetensors)와 어댑터 구성 파일(adapter_config.json)만이 저장

## Datasets

- 주어진 명령어를 이해하고 따르는 '명령어 기반 모델(Instruction-Tuned Model)'
- 명령어 기반 모델은 ...와 같은 특정 대화 마커를 사용자 요청으로, 그 뒤에 오는 텍스트를 모델의 기대 응답으로 연관 짓도록 학습
- SFTTrainer는 단일 text 열(column)로 형식화된 데이터셋을 처리할 수 있도록 설계
- 명령어, 입력, 출력을 명확하게 구분

```prompt
<s> 다음 영문 텍스트를 한국어로 번역하세요: {english_text} {korean_text}</s>
```

## Test using MPS

- Metal Performance Shaders (MPS)
- QLoRA(Quantized Low-Rank Adaptation)
- bitsandbytes or optimum-quanto
- SFTTrainer
- model : google/gemma-3-1b-it
- dataset : lemon-mint/korean_english_parallel_wiki_augmented_v1

test_m4_mps.py

```bash
python test_m4_mps.py 
```

```bash
PyTorch Version: 2.9.0.dev20250810
MPS backend is available.
Successfully created a tensor on the MPS device:
tensor([1.], device='mps:0')
```

load_dataset1.py

[Dataset Preparation](https://mlflow.org/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft/#2-dataset-preparation)

```bash
README.md: 1.78kB [00:00, 4.44MB/s]
data/train-00000-of-00002.parquet: 100%|█| 285M/285M [00:20<00:00, 13.
data/train-00001-of-00002.parquet: 100%|█| 285M/285M [00:22<00:00, 12.
Generating train split: 100%|█| 503245/503245 [00:00<00:00, 790184.93 
Map: 100%|██████████| 503245/503245 [00:07<00:00, 71489.15 examples/s]
훈련 데이터셋 크기: 478082
검증 데이터셋 크기: 25163

변환된 데이터 샘플:
Column(['<s> 다음 영문 텍스트를 한국어로 번역하세요: After the war, Boeselager\'s part in the failed attempt on Hitler\'s life became known. He was regarded as a hero by many in Germany and France and received the highest military medals that both countries could provide. He studied economics and became a forestry expert. Even in his old age, Boeselager still had nightmares about the conspiracy and the friends whom he lost during the war. He urged young people to become more involved in politics, as he felt apathy and the political inexperience of the German masses had been two of the key reasons why Hitler was able to come to power. The entrance to his residence in Kreuzberg bears the Latin motto Et si omnes ego non ("even if all, not I").\n\nBoeselager was a member of K.D.St.V. Ripuaria Bonn, a Roman Catholic student fraternity at the University of Bonn that now belongs to the Cartellverband der katholischen deutschen Studentenverbindungen. Until his death on 1 May 2008, he still had the Walther PP pistol he was given to use to shoot Hitler. 폰 보젤라거는 전쟁 후 히틀러 암살 시도에 대한 그의 역할이 알려졌습니다. 그는 독일과 프랑스에서 많은 사람들에게 영웅으로 여겨졌으며 두 나라에서 제공할 수 있는 최고의 군사 훈장을 받았습니다. 그는 경제학을 공부하고 임업 전문가가 되었습니다. 노년에도 불구하고 폰 보젤라거는 여전히 음모와 전쟁 중에 잃은 친구들을 꿈꾸었습니다. 그는 젊은이들이 정치에 더 적극적으로 참여해야 한다고 촉구했습니다. 그는 독일 대중의 무관심과 정치적 경험 부족이 히틀러가 권력을 잡을 수 있었던 주요 이유 중 두 가지라고 생각했습니다. 크로이츠베르크에 있는 그의 주택 입구에는 "모두가 그렇다고 해도, 나는 그렇지 않다"라는 라틴어 모토인 "Et si omnes ego non"이 새겨져 있습니다.\n\n폰 보젤라거는 본 대학교의 로마 가톨릭 학생 사교 클럽인 K.D.St.V. Ripuaria 본의 회원이었으며, 현재는 카톨릭 독일 학생 연맹인 Cartellverband der katholischen deutschen Studentenverbindungen에 속해 있습니다. 그는 2008년 5월 1일에 사망할 때까지 히틀러를 쏘기 위해 지급받았던 발터 PP 권총을 소유하고 있었습니다.</s>', "<s> 다음 영문 텍스트를 한국어로 번역하세요: The National Transport Commission (NTC), previously known as the National Road Transport Commission, is an Australian statutory body created to develop regulatory and operational reform for road, rail and intermodal transport.\n\nUnder Australia's federal system, transport policy and regulatory responsibilities span across Commonwealth, state and territory, and local governments. Differences between these regulatory systems mean that national transport operators can face inconsistent regulations, creating unnecessary inefficiency and cost.\n\nThe NTC is focused on reforms which improve the productivity, safety and environmental outcomes of the Australian transport system. Its work includes delivering a National Rail Action Plan to make rail more interoperable, developing an end-to-end regulatory framework for automated vehicles, improving the National Heavy Vehicle law, maintaining the Australian Dangerous Goods Code and maintaining national model laws, including the Australian Road Rules. 호주 국립운송위원회(NTC)는 이전에 국립도로운송위원회로 알려졌으며, 도로, 철도 및 연계 운송에 대한 규제 및 운영 개혁을 개발하기 위해 설립된 호주 법정 기구입니다.\n\n호주의 연방 시스템 하에서 운송 정책 및 규제 책임은 연방, 주 및 준주, 지방 정부에 걸쳐 있습니다. 이러한 규제 시스템 간의 차이로 인해 전국 운송 사업자는 불일치하는 규정에 직면하여 불필요한 비효율성과 비용을 초래할 수 있습니다.\n\nNTC는 호주 운송 시스템의 생산성, 안전 및 환경 결과를 개선하는 개혁에 중점을 두고 있습니다. 그들의 작업에는 철도의 상호 운용성을 높이기 위한 국립 철도 행동 계획 제공, 자동 차량에 대한 종단 간 규제 체계 개발, 국립 중량 화물 차량 법 개선, 호주 위험물 규정 유지 및 호주 도로 규칙을 포함한 국가 모델 법 유지 등이 포함됩니다.</s>", "<s> 다음 영문 텍스트를 한국어로 번역하세요: The Medical Women's Association of Nigeria (MWAN) is a Nigerian women's health organization that represents female doctors registered with the Medical and Dental Council of Nigeria (MDCN). The group's mission is to improve women's health in Nigeria through patient advocacy, including offering community health screening programs.\n\nMWAN is affiliated with the Nigerian Medical Association and Medical Women's International Association (MWIA), and operates in 34 of the country's 36 states. 2019 president Joyce Barber handed over, on Saturday, 14 September 2019 to Dr. Mininim I. Oseji, who then emerged as the 22nd National President of the Association. 나이지리아 여성 의사 협회 (MWAN)는 나이지리아 의료 및 치과위원회 (MDCN)에 등록된 여성 의사를 대표하는 나이지리아 여성 건강 기구입니다. 이 단체의 사명은 환자 옹호를 통해 나이지리아 여성의 건강을 개선하는 것입니다. 여기에는 지역 사회 건강 검진 프로그램 제공이 포함됩니다.\n\nMWAN은 나이지리아 의사 협회와 국제 여성 의사 협회 (MWIA)와 제휴하고 있으며 나이지리아 36개 주 중 34개 주에서 활동하고 있습니다. 2019년 회장 조이스 바버는 2019년 9월 14일 토요일에 미니님 아이 오세지 박사에게 회장직을 이임했으며, 오세지 박사는 협회의 22대 회장으로 취임했습니다.</s>", '<s> 다음 영문 텍스트를 한국어로 번역하세요: Throughout their career the brothers\' camera of choice was the Underwood Instanto, which recorded images on 8.5 x 6.5 inch photographic plates. Many of their climbing photographs, (including the classic portrait of Owen Glynne Jones), were reproduced in Alan Hankinson\'s Camera on the Crags. A large selection is also in the possession of the FRCC (The Fell and Rock Climbing Club of the English Lake District), of which the brothers were founding members and Ashley its first president.\n\nThe Abrahams\' photographic shop in Keswick, built in 1887, was taken over in due course by local mountaineer George Fisher; the modern shop still contains many memorabilia, including photographs, from the Abrahams\' era. 케스윅 형제는 등반 경력 내내 언더우드 인스턴토 카메라를 사용했습니다. 이 카메라는 8.5 x 6.5 인치 사진판에 이미지를 기록했습니다. 이들의 등반 사진 중 다수 (오웬 글린 존스의 클래식 초상화 포함)는 앨런 행킨슨의 "암벽 위의 카메라"에 재현되었습니다. 또한 케스윅 형제가 창립 멤버였던 영국 레이크 디스트릭트 펠 & 록 클라이밍 클럽 (FRCC)에 소장되어 있습니다. 애슐리 아브라함은 이 클럽의 초대 회장이었습니다.\n\n케스윅 형제의 사진 가게는 1887년에 지어졌고, 나중에 지역 산악인 조지 피셔에게 인수되었습니다. 현재 가게에는 아브라함 시대의 사진을 포함한 많은 기념품이 전시되어 있습니다.</s>', '<s> 다음 영문 텍스트를 한국어로 번역하세요: The Philippine Civil Code is a comprehensive law governing family and property relations. The influence of the Spanish Civil Code is most evident in the books on property, succession and obligations and contracts. The law on succession, for example, retains such concepts indigenous to Spain such as the rule on legitimes and reserva troncal. On the other hand, many of the provisions on special contracts, particularly on sales, are derived from common law as practised in the United States, reflecting the influence of American colonial rule and the influx of commercial relations involving Americans at the time.\n\nThe great mass of disputes between private persons over civil and property relations are resolved by applying the provisions of the Civil Code. With over 2,000 specific provisions, the Civil Code attempts to anticipate all possible questions arising from civil and property relations and prescribe a definitive solution for these problems. Understandably, the Civil Code itself is unable to provide a definite answer for all emerging problems; thus the courts also rely on precedent based on interpretations by the Supreme Court. This the Civil Code itself notably recognises in saying that "[j]udicial decisions applying or interpreting the laws or the Constitution shall form a part of the legal system of the Philippines" (Article 8, Civil Code), a recognition of the eminent role now played by precedents in Philippine law. The Civil Code is divided into four "books", with each specific book namely: 필리핀 민법은 가족 및 재산 관계를 규율하는 포괄적인 법률입니다. 스페인 민법의 영향을 가장 크게 받은 부분은 재산, 상속 및 채무와 계약에 관한 부분입니다. 예를 들어, 상속에 관한 법률은 합법적인 권리와 혈통 보존 규칙과 같이 스페인에 고유한 개념을 유지하고 있습니다. 반면, 특수 계약, 특히 매매에 관한 많은 조항은 미국에서 시행되는 영미법에서 유래했습니다. 이는 미국의 식민 지배와 당시 미국과의 상업 관계 증가의 영향을 반영합니다.\n\n필리핀 민법은 사적 관계와 재산 관계에 대한 개인 간의 분쟁 대부분을 해결합니다. 2,000개가 넘는 구체적인 조항을 통해 필리핀 민법은 민법 및 재산 관계에서 발생할 수 있는 모든 가능한 문제를 예상하고 이러한 문제에 대한 명확한 해결책을 제시하려고 합니다. 당연히 필리핀 민법 자체로 모든 새로운 문제에 대한 명확한 답을 제공할 수는 없습니다. 따라서 법원은 대법원의 해석을 기반으로 한 선례에 의존합니다. 필리핀 민법은 "법 또는 헌법을 적용하거나 해석하는 사법 판결은 필리핀의 법 체계의 일부를 구성한다"고 명시하여 선례가 필리핀 법에서 중요한 역할을 한다는 것을 인정합니다. 필리핀 민법은 "인물과 가족 관계", "재산, 소유권 및 그 변경", "채무와 계약", "불법 행위와 손해"의 네 가지 책으로 나뉩니다.</s>'])
```

train_save.py

```bash
ValueError: Your setup doesn't support bf16/gpu.

오류 해결:
    #bf16=True, # bfloat16 활성화
    bf16=False, # bfloat16 비활성화
```

```bash
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/load_model.py", line 91, in <module>
    trainer = SFTTrainer(
              ^^^^^^^^^^^
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'

오류 해결:
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    #tokenizer=tokenizer,
    args=training_arguments,
)

참조 사이트:
https://huggingface.co/docs/trl/main/en/sft_trainer
```

```bash
Loading checkpoint shards: 100%|████████████████████████████████| 2/2 [00:09<00:00,  4.61s/it]
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:453: UserWarning: Padding-free training is enabled, but the attention implementation is not set to 'flash_attention_2'. Padding-free training flattens batches into a single sequence, and 'flash_attention_2' is the only known attention mechanism that reliably supports this. Using other implementations may lead to unexpected behavior. To ensure compatibility, set `attn_implementation='flash_attention_2'` in the model configuration, or verify that your attention mechanism can handle flattened sequences.
  warnings.warn(
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:495: UserWarning: You are using packing, but the attention implementation is not set to 'flash_attention_2' or 'kernels-community/vllm-flash-attn3'. Packing flattens batches into a single sequence, and Flash Attention is the only known attention mechanisms that reliably support this. Using other implementations may lead to cross-contamination between batches. To avoid this, either disable packing by setting `packing=False`, or set `attn_implementation='flash_attention_2'` or `attn_implementation='kernels-community/vllm-flash-attn3'` in the model configuration.
  warnings.warn(
Adding EOS to train dataset: 100%|██████████| 478082/478082 [00:15<00:00, 31002.48 examples/s]
Tokenizing train dataset: 100%|██████████████| 478082/478082 [02:55<00:00, 2726.82 examples/s]
Packing train dataset: 100%|███████████████| 478082/478082 [00:02<00:00, 161070.37 examples/s]
Adding EOS to eval dataset: 100%|██████████████| 25163/25163 [00:03<00:00, 6987.33 examples/s]
Tokenizing eval dataset: 100%|█████████████████| 25163/25163 [00:09<00:00, 2660.16 examples/s]
Packing eval dataset: 100%|██████████████████| 25163/25163 [00:00<00:00, 180819.20 examples/s]
  0%|                                                                 | 0/500 [00:00<?, ?it/s]/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
```

```bash
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:453: UserWarning: Padding-free training is enabled, but the attention implementation is not set to 'flash_attention_2'. Padding-free training flattens batches into a single sequence, and 'flash_attention_2' is the only known attention mechanism that reliably supports this. Using other implementations may lead to unexpected behavior. To ensure compatibility, set `attn_implementation='flash_attention_2'` in the model configuration, or verify that your attention mechanism can handle flattened sequences.
  warnings.warn(
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:495: UserWarning: You are using packing, but the attention implementation is not set to 'flash_attention_2' or 'kernels-community/vllm-flash-attn3'. Packing flattens batches into a single sequence, and Flash Attention is the only known attention mechanisms that reliably support this. Using other implementations may lead to cross-contamination between batches. To avoid this, either disable packing by setting `packing=False`, or set `attn_implementation='flash_attention_2'` or `attn_implementation='kernels-community/vllm-flash-attn3'` in the model configuration.
  warnings.warn(
Adding EOS to train dataset: 100%|██████████| 478082/478082 [00:13<00:00, 35554.73 examples/s]
Tokenizing train dataset: 100%|██████████████| 478082/478082 [02:50<00:00, 2797.79 examples/s]
Packing train dataset: 100%|███████████████| 478082/478082 [00:02<00:00, 159635.28 examples/s]
Adding EOS to eval dataset: 100%|██████████████| 25163/25163 [00:03<00:00, 7644.96 examples/s]
Tokenizing eval dataset: 100%|█████████████████| 25163/25163 [00:08<00:00, 2879.77 examples/s]
Packing eval dataset: 100%|██████████████████| 25163/25163 [00:00<00:00, 173992.84 examples/s]
  0%|                                                                 | 0/500 [00:00<?, ?it/s]/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `sdpa`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`.
{'loss': 2.4506, 'grad_norm': 1.0841598510742188, 'learning_rate': 9.82e-05, 'num_tokens': 80276.0, 'mean_token_accuracy': 0.5479228924959898, 'epoch': 0.0}
{'loss': 2.1745, 'grad_norm': 0.7756809592247009, 'learning_rate': 9.620000000000001e-05, 'num_tokens': 160160.0, 'mean_token_accuracy': 0.5698625419288874, 'epoch': 0.0}
{'loss': 2.0378, 'grad_norm': 0.6837954521179199, 'learning_rate': 9.42e-05, 'num_tokens': 240478.0, 'mean_token_accuracy': 0.5893009610474109, 'epoch': 0.0}
{'loss': 2.0136, 'grad_norm': 0.6906253695487976, 'learning_rate': 9.22e-05, 'num_tokens': 320329.0, 'mean_token_accuracy': 0.5916205361485481, 'epoch': 0.0}
{'loss': 1.9448, 'grad_norm': 0.6286749839782715, 'learning_rate': 9.020000000000001e-05, 'num_tokens': 399162.0, 'mean_token_accuracy': 0.598753745853901, 'epoch': 0.0}
{'loss': 1.9811, 'grad_norm': 0.7483819127082825, 'learning_rate': 8.82e-05, 'num_tokens': 479163.0, 'mean_token_accuracy': 0.5942541971802712, 'epoch': 0.0}
{'loss': 1.8907, 'grad_norm': 0.6943105459213257, 'learning_rate': 8.620000000000001e-05, 'num_tokens': 559982.0, 'mean_token_accuracy': 0.6068649165332317, 'epoch': 0.0}
{'loss': 1.895, 'grad_norm': 0.6525490880012512, 'learning_rate': 8.42e-05, 'num_tokens': 639281.0, 'mean_token_accuracy': 0.6077428877353668, 'epoch': 0.0}
{'loss': 1.852, 'grad_norm': 0.6052327752113342, 'learning_rate': 8.22e-05, 'num_tokens': 719430.0, 'mean_token_accuracy': 0.6114868529140949, 'epoch': 0.0}
{'loss': 1.8899, 'grad_norm': 0.6816254258155823, 'learning_rate': 8.020000000000001e-05, 'num_tokens': 798670.0, 'mean_token_accuracy': 0.6077064961194992, 'epoch': 0.0}
{'loss': 1.9168, 'grad_norm': 0.7224716544151306, 'learning_rate': 7.82e-05, 'num_tokens': 879346.0, 'mean_token_accuracy': 0.602442742139101, 'epoch': 0.0}
{'loss': 1.906, 'grad_norm': 0.704435408115387, 'learning_rate': 7.620000000000001e-05, 'num_tokens': 960294.0, 'mean_token_accuracy': 0.6027944058179855, 'epoch': 0.0}
{'loss': 1.8401, 'grad_norm': 0.7145906090736389, 'learning_rate': 7.42e-05, 'num_tokens': 1040472.0, 'mean_token_accuracy': 0.6138291202485562, 'epoch': 0.0}
{'loss': 1.8496, 'grad_norm': 0.738406240940094, 'learning_rate': 7.22e-05, 'num_tokens': 1120168.0, 'mean_token_accuracy': 0.614564336836338, 'epoch': 0.01}
{'loss': 1.8652, 'grad_norm': 0.6938950419425964, 'learning_rate': 7.02e-05, 'num_tokens': 1200518.0, 'mean_token_accuracy': 0.611142023652792, 'epoch': 0.01}
{'loss': 1.8291, 'grad_norm': 0.6831591725349426, 'learning_rate': 6.82e-05, 'num_tokens': 1281425.0, 'mean_token_accuracy': 0.6156429432332515, 'epoch': 0.01}
{'loss': 1.8526, 'grad_norm': 0.7474468946456909, 'learning_rate': 6.620000000000001e-05, 'num_tokens': 1360759.0, 'mean_token_accuracy': 0.6082599937915802, 'epoch': 0.01}
{'loss': 1.8345, 'grad_norm': 0.7025880813598633, 'learning_rate': 6.42e-05, 'num_tokens': 1440287.0, 'mean_token_accuracy': 0.6133685585111379, 'epoch': 0.01}
{'loss': 1.8221, 'grad_norm': 0.7364481687545776, 'learning_rate': 6.220000000000001e-05, 'num_tokens': 1520658.0, 'mean_token_accuracy': 0.618384450674057, 'epoch': 0.01}
{'loss': 1.8152, 'grad_norm': 0.7253894805908203, 'learning_rate': 6.02e-05, 'num_tokens': 1600509.0, 'mean_token_accuracy': 0.619917419552803, 'epoch': 0.01}
{'loss': 1.8163, 'grad_norm': 0.7602978944778442, 'learning_rate': 5.82e-05, 'num_tokens': 1680565.0, 'mean_token_accuracy': 0.616064828634262, 'epoch': 0.01}
{'loss': 1.8383, 'grad_norm': 0.702643871307373, 'learning_rate': 5.620000000000001e-05, 'num_tokens': 1760847.0, 'mean_token_accuracy': 0.6105592761188745, 'epoch': 0.01}
{'loss': 1.7947, 'grad_norm': 0.7568259239196777, 'learning_rate': 5.420000000000001e-05, 'num_tokens': 1840587.0, 'mean_token_accuracy': 0.6194721885025501, 'epoch': 0.01}
{'loss': 1.8139, 'grad_norm': 0.8016412854194641, 'learning_rate': 5.22e-05, 'num_tokens': 1919975.0, 'mean_token_accuracy': 0.6183466047048569, 'epoch': 0.01}
{'loss': 1.8366, 'grad_norm': 0.7635400891304016, 'learning_rate': 5.02e-05, 'num_tokens': 2000193.0, 'mean_token_accuracy': 0.6137465000152588, 'epoch': 0.01}
{'loss': 1.8047, 'grad_norm': 0.764035165309906, 'learning_rate': 4.82e-05, 'num_tokens': 2080223.0, 'mean_token_accuracy': 0.6151143297553062, 'epoch': 0.01}
{'loss': 1.7906, 'grad_norm': 0.7395619750022888, 'learning_rate': 4.6200000000000005e-05, 'num_tokens': 2159164.0, 'mean_token_accuracy': 0.6183772653341293, 'epoch': 0.01}
{'loss': 1.7912, 'grad_norm': 0.8469703197479248, 'learning_rate': 4.4200000000000004e-05, 'num_tokens': 2238903.0, 'mean_token_accuracy': 0.621285579353571, 'epoch': 0.01}
{'loss': 1.8169, 'grad_norm': 0.8273006081581116, 'learning_rate': 4.22e-05, 'num_tokens': 2319933.0, 'mean_token_accuracy': 0.6175973318517208, 'epoch': 0.01}
{'loss': 1.8321, 'grad_norm': 0.8653826713562012, 'learning_rate': 4.02e-05, 'num_tokens': 2399505.0, 'mean_token_accuracy': 0.6139784328639507, 'epoch': 0.01}
{'loss': 1.8167, 'grad_norm': 0.77748703956604, 'learning_rate': 3.82e-05, 'num_tokens': 2480200.0, 'mean_token_accuracy': 0.6150427110493183, 'epoch': 0.01}
{'loss': 1.7867, 'grad_norm': 0.769636869430542, 'learning_rate': 3.62e-05, 'num_tokens': 2560855.0, 'mean_token_accuracy': 0.6227942690253258, 'epoch': 0.01}
{'loss': 1.76, 'grad_norm': 0.7414308786392212, 'learning_rate': 3.4200000000000005e-05, 'num_tokens': 2640713.0, 'mean_token_accuracy': 0.6267132334411144, 'epoch': 0.01}
{'loss': 1.841, 'grad_norm': 0.7138774394989014, 'learning_rate': 3.2200000000000003e-05, 'num_tokens': 2720857.0, 'mean_token_accuracy': 0.6119803778827191, 'epoch': 0.01}
{'loss': 1.8468, 'grad_norm': 0.8339102864265442, 'learning_rate': 3.02e-05, 'num_tokens': 2801350.0, 'mean_token_accuracy': 0.6123791374266148, 'epoch': 0.01}
{'loss': 1.7754, 'grad_norm': 0.8745675086975098, 'learning_rate': 2.8199999999999998e-05, 'num_tokens': 2880865.0, 'mean_token_accuracy': 0.6244201429188252, 'epoch': 0.01}
{'loss': 1.7997, 'grad_norm': 0.8256745934486389, 'learning_rate': 2.6200000000000003e-05, 'num_tokens': 2959764.0, 'mean_token_accuracy': 0.6188802801072597, 'epoch': 0.01}
{'loss': 1.8034, 'grad_norm': 0.7599915862083435, 'learning_rate': 2.4200000000000002e-05, 'num_tokens': 3040176.0, 'mean_token_accuracy': 0.6176834151148796, 'epoch': 0.01}
{'loss': 1.7983, 'grad_norm': 0.727372944355011, 'learning_rate': 2.22e-05, 'num_tokens': 3119846.0, 'mean_token_accuracy': 0.6177359625697136, 'epoch': 0.01}
{'loss': 1.8031, 'grad_norm': 0.7672078609466553, 'learning_rate': 2.0200000000000003e-05, 'num_tokens': 3200365.0, 'mean_token_accuracy': 0.6185009308159352, 'epoch': 0.01}
{'loss': 1.7719, 'grad_norm': 0.7751895189285278, 'learning_rate': 1.8200000000000002e-05, 'num_tokens': 3280631.0, 'mean_token_accuracy': 0.6248038351535797, 'epoch': 0.02}
{'loss': 1.782, 'grad_norm': 0.8023542165756226, 'learning_rate': 1.62e-05, 'num_tokens': 3361520.0, 'mean_token_accuracy': 0.6228289715945721, 'epoch': 0.02}
{'loss': 1.7679, 'grad_norm': 0.7920138835906982, 'learning_rate': 1.42e-05, 'num_tokens': 3441382.0, 'mean_token_accuracy': 0.6222206756472588, 'epoch': 0.02}
{'loss': 1.7813, 'grad_norm': 0.7416978478431702, 'learning_rate': 1.22e-05, 'num_tokens': 3520875.0, 'mean_token_accuracy': 0.6233970317989588, 'epoch': 0.02}
{'loss': 1.7558, 'grad_norm': 0.7693504691123962, 'learning_rate': 1.02e-05, 'num_tokens': 3601056.0, 'mean_token_accuracy': 0.627298029512167, 'epoch': 0.02}
{'loss': 1.7414, 'grad_norm': 0.7517438530921936, 'learning_rate': 8.200000000000001e-06, 'num_tokens': 3681197.0, 'mean_token_accuracy': 0.6275794960558414, 'epoch': 0.02}
{'loss': 1.8101, 'grad_norm': 0.7494860291481018, 'learning_rate': 6.2e-06, 'num_tokens': 3761381.0, 'mean_token_accuracy': 0.616555942222476, 'epoch': 0.02}
{'loss': 1.774, 'grad_norm': 0.754429817199707, 'learning_rate': 4.2000000000000004e-06, 'num_tokens': 3841541.0, 'mean_token_accuracy': 0.625054232776165, 'epoch': 0.02}
{'loss': 1.7775, 'grad_norm': 0.7748628854751587, 'learning_rate': 2.2e-06, 'num_tokens': 3921514.0, 'mean_token_accuracy': 0.6213903181254864, 'epoch': 0.02}
{'loss': 1.7757, 'grad_norm': 0.7851539254188538, 'learning_rate': 2.0000000000000002e-07, 'num_tokens': 4001852.0, 'mean_token_accuracy': 0.6244115874171257, 'epoch': 0.02}
{'train_runtime': 32806.1048, 'train_samples_per_second': 0.122, 'train_steps_per_second': 0.015, 'train_loss': 1.8512983055114747, 'epoch': 0.02}   
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [9:06:46<00:00, 65.61s/it]
```

```bash
It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `sdpa`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`.

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="eager",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16, # bfloat16 사용으로 성능 향상 기대
    device_map="mps"
)

설정 변경으로 동작 속도가 빨라짐.
```

```bash
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:453: UserWarning: Padding-free training is enabled, but the attention implementation is not set to 'flash_attention_2'. Padding-free training flattens batches into a single sequence, and 'flash_attention_2' is the only known attention mechanism that reliably supports this. Using other implementations may lead to unexpected behavior. To ensure compatibility, set `attn_implementation='flash_attention_2'` in the model configuration, or verify that your attention mechanism can handle flattened sequences.
  warnings.warn(
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:495: UserWarning: You are using packing, but the attention implementation is not set to 'flash_attention_2' or 'kernels-community/vllm-flash-attn3'. Packing flattens batches into a single sequence, and Flash Attention is the only known attention mechanisms that reliably support this. Using other implementations may lead to cross-contamination between batches. To avoid this, either disable packing by setting `packing=False`, or set `attn_implementation='flash_attention_2'` or `attn_implementation='kernels-community/vllm-flash-attn3'` in the model configuration.
  warnings.warn(
Adding EOS to train dataset: 100%|█████████████████████████████████| 478082/478082 [00:13<00:00, 36314.41 examples/s]
Tokenizing train dataset: 100%|█████████████████████████████████████| 478082/478082 [03:28<00:00, 2294.33 examples/s]
Packing train dataset: 100%|██████████████████████████████████████| 478082/478082 [00:02<00:00, 160165.93 examples/s]
Adding EOS to eval dataset: 100%|█████████████████████████████████████| 25163/25163 [00:03<00:00, 8171.26 examples/s]
Tokenizing eval dataset: 100%|████████████████████████████████████████| 25163/25163 [00:11<00:00, 2285.75 examples/s]
Packing eval dataset: 100%|█████████████████████████████████████████| 25163/25163 [00:00<00:00, 175119.05 examples/s]
  0%|                                                                                        | 0/500 [00:00<?, ?it/s]/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 2.4266, 'grad_norm': 1.0490840673446655, 'learning_rate': 9.82e-05, 'num_tokens': 80010.0, 'mean_token_accuracy': 0.5522503785789012, 'epoch': 0.0}
{'loss': 2.1895, 'grad_norm': 1.6383039951324463, 'learning_rate': 9.620000000000001e-05, 'num_tokens': 159946.0, 'mean_token_accuracy': 0.5667015470564365, 'epoch': 0.0}
{'loss': 2.0425, 'grad_norm': 0.8705893158912659, 'learning_rate': 9.42e-05, 'num_tokens': 239271.0, 'mean_token_accuracy': 0.5879059351980687, 'epoch': 0.0}
{'loss': 1.9556, 'grad_norm': 0.6672195792198181, 'learning_rate': 9.22e-05, 'num_tokens': 319443.0, 'mean_token_accuracy': 0.5978349901735782, 'epoch': 0.0}
{'loss': 1.9962, 'grad_norm': 0.695339560508728, 'learning_rate': 9.020000000000001e-05, 'num_tokens': 399106.0, 'mean_token_accuracy': 0.5945642828941345, 'epoch': 0.0}
{'loss': 1.9229, 'grad_norm': 0.6518824696540833, 'learning_rate': 8.82e-05, 'num_tokens': 479588.0, 'mean_token_accuracy': 0.6019796214997768, 'epoch': 0.0}
{'loss': 1.9206, 'grad_norm': 0.7191448211669922, 'learning_rate': 8.620000000000001e-05, 'num_tokens': 560074.0, 'mean_token_accuracy': 0.6018661379814148, 'epoch': 0.0}
{'loss': 1.906, 'grad_norm': 0.7719150185585022, 'learning_rate': 8.42e-05, 'num_tokens': 640689.0, 'mean_token_accuracy': 0.6055508714169264, 'epoch': 0.0}
{'loss': 1.9367, 'grad_norm': 0.666132390499115, 'learning_rate': 8.22e-05, 'num_tokens': 720807.0, 'mean_token_accuracy': 0.5989571116864681, 'epoch': 0.0}
{'loss': 1.8829, 'grad_norm': 0.6830726265907288, 'learning_rate': 8.020000000000001e-05, 'num_tokens': 800463.0, 'mean_token_accuracy': 0.6057536624372005, 'epoch': 0.0}
 20%|██████████████▊                                                           | 100/500 [1:00:03<4:24:09, 39.62s/it]
```

```bash
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:453: UserWarning: Padding-free training is enabled, but the attention implementation is not set to 'flash_attention_2'. Padding-free training flattens batches into a single sequence, and 'flash_attention_2' is the only known attention mechanism that reliably supports this. Using other implementations may lead to unexpected behavior. To ensure compatibility, set `attn_implementation='flash_attention_2'` in the model configuration, or verify that your attention mechanism can handle flattened sequences.
  warnings.warn(
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:495: UserWarning: You are using packing, but the attention implementation is not set to 'flash_attention_2' or 'kernels-community/vllm-flash-attn3'. Packing flattens batches into a single sequence, and Flash Attention is the only known attention mechanisms that reliably support this. Using other implementations may lead to cross-contamination between batches. To avoid this, either disable packing by setting `packing=False`, or set `attn_implementation='flash_attention_2'` or `attn_implementation='kernels-community/vllm-flash-attn3'` in the model configuration.
  warnings.warn(
Adding EOS to train dataset: 100%|█████| 478082/478082 [00:15<00:00, 30223.19 examples/s]
Tokenizing train dataset: 100%|█████████| 478082/478082 [02:41<00:00, 2952.01 examples/s]
Packing train dataset: 100%|██████████| 478082/478082 [00:02<00:00, 171952.91 examples/s]
Adding EOS to eval dataset: 100%|█████████| 25163/25163 [00:02<00:00, 8611.54 examples/s]
Tokenizing eval dataset: 100%|████████████| 25163/25163 [00:08<00:00, 2864.36 examples/s]
Packing eval dataset: 100%|█████████████| 25163/25163 [00:00<00:00, 190734.25 examples/s]
  0%|                                                            | 0/500 [00:00<?, ?it/s]/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 2.4458, 'grad_norm': 1.0792218446731567, 'learning_rate': 9.82e-05, 'num_tokens': 80674.0, 'mean_token_accuracy': 0.5482614107429982, 'epoch': 0.0}
{'loss': 2.2348, 'grad_norm': 1.071425437927246, 'learning_rate': 9.620000000000001e-05, 'num_tokens': 160356.0, 'mean_token_accuracy': 0.5611519511789084, 'epoch': 0.0}
{'loss': 2.0936, 'grad_norm': 0.815075695514679, 'learning_rate': 9.42e-05, 'num_tokens': 240364.0, 'mean_token_accuracy': 0.5789744418114424, 'epoch': 0.0}
{'loss': 2.026, 'grad_norm': 0.6516107320785522, 'learning_rate': 9.22e-05, 'num_tokens': 320708.0, 'mean_token_accuracy': 0.5912419654428959, 'epoch': 0.0}
{'loss': 1.9737, 'grad_norm': 0.6816374063491821, 'learning_rate': 9.020000000000001e-05, 'num_tokens': 400892.0, 'mean_token_accuracy': 0.5949664950370789, 'epoch': 0.0}
{'loss': 1.948, 'grad_norm': 0.6081022024154663, 'learning_rate': 8.82e-05, 'num_tokens': 480450.0, 'mean_token_accuracy': 0.6027136608958245, 'epoch': 0.0}
{'loss': 1.9089, 'grad_norm': 0.6951001286506653, 'learning_rate': 8.620000000000001e-05, 'num_tokens': 560237.0, 'mean_token_accuracy': 0.6042911216616631, 'epoch': 0.0}
{'loss': 1.8905, 'grad_norm': 0.7139027118682861, 'learning_rate': 8.42e-05, 'num_tokens': 640651.0, 'mean_token_accuracy': 0.6042694006115198, 'epoch': 0.0}
{'loss': 1.9027, 'grad_norm': 0.721149742603302, 'learning_rate': 8.22e-05, 'num_tokens': 721398.0, 'mean_token_accuracy': 0.6043803781270981, 'epoch': 0.0}
{'loss': 1.8872, 'grad_norm': 0.6768695712089539, 'learning_rate': 8.020000000000001e-05, 'num_tokens': 801575.0, 'mean_token_accuracy': 0.6066996425390243, 'epoch': 0.0}
{'loss': 1.8891, 'grad_norm': 0.9051889777183533, 'learning_rate': 7.82e-05, 'num_tokens': 881348.0, 'mean_token_accuracy': 0.6104719329625368, 'epoch': 0.0}
{'loss': 1.8588, 'grad_norm': 0.7520777583122253, 'learning_rate': 7.620000000000001e-05, 'num_tokens': 961811.0, 'mean_token_accuracy': 0.6072161436080933, 'epoch': 0.0}
{'loss': 1.8441, 'grad_norm': 0.6979877352714539, 'learning_rate': 7.42e-05, 'num_tokens': 1041799.0, 'mean_token_accuracy': 0.6132455468177795, 'epoch': 0.0}
{'loss': 1.8969, 'grad_norm': 0.8274825811386108, 'learning_rate': 7.22e-05, 'num_tokens': 1121376.0, 'mean_token_accuracy': 0.6033559605479241, 'epoch': 0.01}
{'loss': 1.873, 'grad_norm': 0.8586711883544922, 'learning_rate': 7.02e-05, 'num_tokens': 1200818.0, 'mean_token_accuracy': 0.6085720349103212, 'epoch': 0.01}
{'loss': 1.824, 'grad_norm': 0.7504419684410095, 'learning_rate': 6.82e-05, 'num_tokens': 1280049.0, 'mean_token_accuracy': 0.618630213290453, 'epoch': 0.01}
{'loss': 1.7738, 'grad_norm': 0.7328142523765564, 'learning_rate': 6.620000000000001e-05, 'num_tokens': 1360219.0, 'mean_token_accuracy': 0.624957051128149, 'epoch': 0.01}
{'loss': 1.8861, 'grad_norm': 0.8358198404312134, 'learning_rate': 6.42e-05, 'num_tokens': 1440705.0, 'mean_token_accuracy': 0.6057978380471468, 'epoch': 0.01}
{'loss': 1.86, 'grad_norm': 0.7272653579711914, 'learning_rate': 6.220000000000001e-05, 'num_tokens': 1521852.0, 'mean_token_accuracy': 0.6105055056512356, 'epoch': 0.01}
{'loss': 1.8462, 'grad_norm': 0.765559732913971, 'learning_rate': 6.02e-05, 'num_tokens': 1601428.0, 'mean_token_accuracy': 0.6137678667902946, 'epoch': 0.01}
{'loss': 1.8214, 'grad_norm': 0.7217767834663391, 'learning_rate': 5.82e-05, 'num_tokens': 1681845.0, 'mean_token_accuracy': 0.6170408077538013, 'epoch': 0.01}
{'loss': 1.7925, 'grad_norm': 0.7126762866973877, 'learning_rate': 5.620000000000001e-05, 'num_tokens': 1761919.0, 'mean_token_accuracy': 0.618534979224205, 'epoch': 0.01}
{'loss': 1.8234, 'grad_norm': 0.7211499214172363, 'learning_rate': 5.420000000000001e-05, 'num_tokens': 1841572.0, 'mean_token_accuracy': 0.6177551060914993, 'epoch': 0.01}
{'loss': 1.7842, 'grad_norm': 0.7576083540916443, 'learning_rate': 5.22e-05, 'num_tokens': 1920674.0, 'mean_token_accuracy': 0.6232095703482627, 'epoch': 0.01}
{'loss': 1.8298, 'grad_norm': 0.7907074093818665, 'learning_rate': 5.02e-05, 'num_tokens': 2000541.0, 'mean_token_accuracy': 0.6117887184023857, 'epoch': 0.01}
{'loss': 1.7924, 'grad_norm': 0.749530553817749, 'learning_rate': 4.82e-05, 'num_tokens': 2080590.0, 'mean_token_accuracy': 0.6203194089233875, 'epoch': 0.01}
{'loss': 1.8121, 'grad_norm': 0.6908621788024902, 'learning_rate': 4.6200000000000005e-05, 'num_tokens': 2160594.0, 'mean_token_accuracy': 0.6187204562127591, 'epoch': 0.01}
{'loss': 1.8119, 'grad_norm': 0.7272445559501648, 'learning_rate': 4.4200000000000004e-05, 'num_tokens': 2240945.0, 'mean_token_accuracy': 0.6190164495259524, 'epoch': 0.01}
{'loss': 1.7909, 'grad_norm': 0.8078261613845825, 'learning_rate': 4.22e-05, 'num_tokens': 2320731.0, 'mean_token_accuracy': 0.6229180693626404, 'epoch': 0.01}
{'loss': 1.7906, 'grad_norm': 0.8375656008720398, 'learning_rate': 4.02e-05, 'num_tokens': 2400277.0, 'mean_token_accuracy': 0.6191619679331779, 'epoch': 0.01}
{'loss': 1.7867, 'grad_norm': 0.7808974981307983, 'learning_rate': 3.82e-05, 'num_tokens': 2480767.0, 'mean_token_accuracy': 0.6201421223580837, 'epoch': 0.01}
{'loss': 1.8185, 'grad_norm': 0.7841443419456482, 'learning_rate': 3.62e-05, 'num_tokens': 2560939.0, 'mean_token_accuracy': 0.6177653767168522, 'epoch': 0.01}
{'loss': 1.7744, 'grad_norm': 0.7541474103927612, 'learning_rate': 3.4200000000000005e-05, 'num_tokens': 2640738.0, 'mean_token_accuracy': 0.6206730686128139, 'epoch': 0.01}
{'loss': 1.8433, 'grad_norm': 0.8593119382858276, 'learning_rate': 3.2200000000000003e-05, 'num_tokens': 2721224.0, 'mean_token_accuracy': 0.6131503291428089, 'epoch': 0.01}
{'loss': 1.8112, 'grad_norm': 0.7251067161560059, 'learning_rate': 3.02e-05, 'num_tokens': 2801445.0, 'mean_token_accuracy': 0.6173579320311546, 'epoch': 0.01}
{'loss': 1.7749, 'grad_norm': 0.8683161735534668, 'learning_rate': 2.8199999999999998e-05, 'num_tokens': 2881260.0, 'mean_token_accuracy': 0.6225243508815765, 'epoch': 0.01}
{'loss': 1.7911, 'grad_norm': 0.7571055293083191, 'learning_rate': 2.6200000000000003e-05, 'num_tokens': 2961776.0, 'mean_token_accuracy': 0.621114706248045, 'epoch': 0.01}
{'loss': 1.7756, 'grad_norm': 0.7843670845031738, 'learning_rate': 2.4200000000000002e-05, 'num_tokens': 3041644.0, 'mean_token_accuracy': 0.6222841337323188, 'epoch': 0.01}
{'loss': 1.7982, 'grad_norm': 0.8566311001777649, 'learning_rate': 2.22e-05, 'num_tokens': 3121072.0, 'mean_token_accuracy': 0.6171835482120513, 'epoch': 0.01}
{'loss': 1.7812, 'grad_norm': 0.9980900883674622, 'learning_rate': 2.0200000000000003e-05, 'num_tokens': 3201609.0, 'mean_token_accuracy': 0.6236084267497063, 'epoch': 0.01}
{'loss': 1.7919, 'grad_norm': 0.774206280708313, 'learning_rate': 1.8200000000000002e-05, 'num_tokens': 3280961.0, 'mean_token_accuracy': 0.6217819713056087, 'epoch': 0.02}
{'loss': 1.7785, 'grad_norm': 1.0630640983581543, 'learning_rate': 1.62e-05, 'num_tokens': 3359754.0, 'mean_token_accuracy': 0.6217516481876373, 'epoch': 0.02}
{'loss': 1.7816, 'grad_norm': 0.8605789542198181, 'learning_rate': 1.42e-05, 'num_tokens': 3440233.0, 'mean_token_accuracy': 0.6195840448141098, 'epoch': 0.02}
{'loss': 1.7842, 'grad_norm': 0.8000230193138123, 'learning_rate': 1.22e-05, 'num_tokens': 3520298.0, 'mean_token_accuracy': 0.6219141043722629, 'epoch': 0.02}
{'loss': 1.8141, 'grad_norm': 0.7335033416748047, 'learning_rate': 1.02e-05, 'num_tokens': 3600965.0, 'mean_token_accuracy': 0.6153104595839978, 'epoch': 0.02}
{'loss': 1.8167, 'grad_norm': 0.768768310546875, 'learning_rate': 8.200000000000001e-06, 'num_tokens': 3680065.0, 'mean_token_accuracy': 0.6182617165148259, 'epoch': 0.02}
{'loss': 1.7489, 'grad_norm': 0.7524411082267761, 'learning_rate': 6.2e-06, 'num_tokens': 3760022.0, 'mean_token_accuracy': 0.6254064865410328, 'epoch': 0.02}
{'loss': 1.7662, 'grad_norm': 0.7679119110107422, 'learning_rate': 4.2000000000000004e-06, 'num_tokens': 3840480.0, 'mean_token_accuracy': 0.6252654597163201, 'epoch': 0.02}
{'loss': 1.7342, 'grad_norm': 0.7599970102310181, 'learning_rate': 2.2e-06, 'num_tokens': 3920356.0, 'mean_token_accuracy': 0.6324793502688408, 'epoch': 0.02}
{'loss': 1.7806, 'grad_norm': 0.7588939666748047, 'learning_rate': 2.0000000000000002e-07, 'num_tokens': 4000811.0, 'mean_token_accuracy': 0.6212163098156452, 'epoch': 0.02}
{'train_runtime': 18267.1993, 'train_samples_per_second': 0.219, 'train_steps_per_second': 0.027, 'train_loss': 1.8532886428833009, 'epoch': 0.02}
100%|████████████████████████████████████████████████| 500/500 [5:04:27<00:00, 36.53s/it]
```

generate_from_ft.py

```bash
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
--- 원본 Gemma 3 모델 추론 결과 ---
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/generate_ft.py", line 54, in <module>
    # 5. 비교 추론 실행
                 ^
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/generate_ft.py", line 45, in translate_text
    
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 3896, in decode
    return self._decode(
           ^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/tokenization_utils_fast.py", line 682, in _decode
    text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: argument 'ids': 'list' object cannot be interpreted as an integer
```

소스 코드 수정 - Gemini

```python
...
    #response = tokenizer.decode(outputs, skip_special_tokens=True)
    # 응답에서 프롬프트 부분 제거
    #translated_text = response.split("")[-1].strip()
    
    # [수정점 1] 2D 텐서에서 1D 텐서를 추출합니다. (outputs[0])
    # [수정점 2] 원본 프롬프트를 제외하고, 새로 생성된 토큰들만 디코딩합니다.
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]

    translated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
...
```

```bash
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
--- 원본 Gemma 3 모델 추론 결과 ---
입력 (영어): Fine-tuning a large language model on Apple Silicon is now more accessible than ever.
출력 (한국어):  Apple has made it possible to train a significant number of large language models on its own chip without needing to run these models on the top-tier models on other platforms.

The best way to fine-tune this model with the Apple Silicon, is using the original training data that produced this model, and its own training procedures. Fine-tuning is not necessary to make the model larger.

In this case, the training dataset has been designed to teach the model the tasks it needs to perform such

--- 미세조정된 모델 추론 결과 ---
입력 (영어): Fine-tuning a large language model on Apple Silicon is now more accessible than ever.
출력 (한국어):  After being trained by Google on their own hardware, Google has been working on bringing the model to Apple's platforms and tools.

Apple offers a variety of custom-built tools and APIs (Application Programming Interfaces) that enable developers to leverage the model to build innovative applications and services. These include APIs to make the model create text, generate images, and perform tasks such as chatbots and game makers.

The same large language model can be implemented using Apple's base iOS apps like Safari, Gmail
```

test_translate.py

```bash
'google/gemma-3-1b-it' 모델을 로드합니다. 잠시 기다려 주세요...
모델 로드가 완료되었습니다.

입력 (영어): The current time is Wednesday, August 13, 2025 at 9:34 AM in Seoul, South Korea.
출력 (한국어): Here's a translation of the text into Korean:

**오늘 (Oneul) - Wednesday**
**8월 13일 (Eollmay 13il) - August 13th**
**9:34 AM (Septu-myeok (am) 9:34) - 9:34 AM**
**서울 (Seoul) - Seoul**
**한국 (Hanguk) - South Korea**

**Complete
```

```bash
'google/gemma-3-4b-it' 모델을 로드합니다. 잠시 기다려 주세요...
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████| 2/2 [00:09<00:00,  4.54s/it]
모델 로드가 완료되었습니다.

입력 (영어): The current time is Wednesday, August 13, 2025 at 9:34 AM in Seoul, South Korea.
출력 (한국어): Here are a few options for translating the sentence, with slightly different nuances:

**Option 1 (Most natural and common):**

"현재 시간은 2025년 8월 13일 화요일 오전 9시 34분, 서울, 대한민국입니다."

* **현재 시간은 (hyeon-jae si-gan-eun):** The current time is
* **2025년 8월 13일
```

```bash
'google/gemma-3-12b-it' 모델을 로드합니다. 잠시 기다려 주세요...
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████| 916/916 [00:00<00:00, 1.09MB/s]
model.safetensors.index.json: 100%|████████████████████████████████████████████████████████████████| 109k/109k [00:00<00:00, 11.7MB/s]
model-00005-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████| 4.60G/4.60G [23:26<00:00, 3.27MB/s]
model-00001-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████| 4.98G/4.98G [24:27<00:00, 3.39MB/s]
model-00002-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████| 4.93G/4.93G [25:15<00:00, 3.25MB/s]
model-00003-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████| 4.93G/4.93G [25:17<00:00, 3.25MB/s]
model-00004-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████| 4.93G/4.93G [25:22<00:00, 3.24MB/s]
Fetching 5 files: 100%|████████████████████████████████████████████████████████████████████████████████| 5/5 [25:23<00:00, 304.68s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.48s/it]
generation_config.json: 100%|████████████████████████████████████████████████████████████████████████| 215/215 [00:00<00:00, 1.92MB/s]
Some parameters are on the meta device because they were offloaded to the disk.
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████| 1.16M/1.16M [00:00<00:00, 4.51MB/s]
tokenizer.model: 100%|███████████████████████████████████████████████████████████████████████████| 4.69M/4.69M [00:01<00:00, 2.68MB/s]
tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████| 33.4M/33.4M [00:02<00:00, 13.9MB/s]
added_tokens.json: 100%|████████████████████████████████████████████████████████████████████████████| 35.0/35.0 [00:00<00:00, 187kB/s]
special_tokens_map.json: 100%|███████████████████████████████████████████████████████████████████████| 662/662 [00:00<00:00, 2.43MB/s]
모델 로드가 완료되었습니다.

입력 (영어): The current time is Wednesday, August 13, 2025 at 9:34 AM in Seoul, South Korea.
출력 (한국어): Here are a few options for translating the sentence, depending on the level of formality you want:

**Option 1 (Most Common & Neutral):**

"지금 서울, 대한민국 시간은 2025년 8월 13일 수요일 오전 9시 34분입니다."

*   **지금 (jigeum):** Now, currently
*   **서울, 대한민국 시간은 (Seoul, Daehanmingguk siganeun):**
```

```bash
'google/gemma-3-1b-it' 모델을 로드합니다. 잠시 기다려 주세요...
모델 로드가 완료되었습니다.

입력 (영어): The current time is Wednesday, August 13, 2025 at 9:34 AM in Seoul, South Korea.
출력 (한국어): Here's the translation of the text into Korean:

**"오늘, 8월 13일 9시 34분 서울, 대한민국에 있습니다."**

This translates directly to: "Today, August 13th, 9:34 AM in Seoul, South Korea."

Is there anything else you'd like me to translate, or perhaps you'd like a slightly different phrasing?
총 소요 시간: 77.14 초
```

```bash
'google/gemma-3-4b-it' 모델을 로드합니다. 잠시 기다려 주세요...
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.86s/it]
모델 로드가 완료되었습니다.

입력 (영어): The current time is Wednesday, August 13, 2025 at 9:34 AM in Seoul, South Korea.
출력 (한국어): Here are a few options for the translation, with slightly different nuances:

**Option 1 (Most common and natural):**

"현재 시간은 2025년 8월 13일 화요일 오전 9시 34분, 서울, 대한민국입니다."

* **현재 시간은 (hyeon-jae si-gan-eun):** The current time is
* **2025년 8월 13일 (
총 소요 시간: 119.48 초
```

```bash
'google/gemma-3-12b-it' 모델을 로드합니다. 잠시 기다려 주세요...
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████| 5/5 [00:23<00:00,  4.69s/it]
모델 로드가 완료되었습니다.

입력 (영어): The current time is Wednesday, August 13, 2025 at 9:34 AM in Seoul, South Korea.
출력 (한국어): Here are a few options for translating "The current time is Wednesday, August 13, 2025 at 9:34 AM in Seoul, South Korea," depending on the level of formality you desire:

**Option 1 (Formal & Complete):**

"현재 시간은 2025년 8월 13일 수요일 오전 9시 34분, 대한민국 서울입니다."

*   **현재 시간은 (hyeon
총 소요 시간: 321.52 초
```