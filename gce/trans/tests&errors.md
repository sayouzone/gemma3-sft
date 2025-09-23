#

| vCPU | Memory | Model | Test | Step |
| --- | --- | --- | --- | --- |
| 4(Core 2) | 32GB | Gemma-3-4b | 47:42 / 3.13s/it | Finetune |
| 4(Core 2) | 32GB | Gemma-3-12b | 1:53:51 / 7.47s/it | Finetune |
| 4(Core 2) | 32GB | Gemma-3-4b | 0:34 / 17.15s/it | Merge |
| 4(Core 2) | 32GB | Gemma-3-12b |  /  | Merge |
| 8(Core 4) | 54GB | Gemma-3-12b |  /  | Merge |
| 16(Core 8) | 64GB | Gemma-3-12b | 01:23 / 16.67s/it | Merge |
| 4(Core 2) | 32GB | Gemma-3-4b | 답변 완료 | Inference |
| 4(Core 2) | 32GB | Gemma-3-12b | Failed | Inference |
| 8(Core 16) | 54GB | Gemma-3-12b | Failed | Inference |
| 16(Core 32) | 64GB | Gemma-3-12b |  | Inference |

##

```bash
python3 finetune_trans.py --method finetune  \
  --model gemma-3-4b \
  --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv \
  --output trans-en-ko

python3 finetune_trans.py --method merge \
  --model gemma-3-4b \
  --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv \
  --output trans-en-ko

python3 finetune_trans.py --method finetune \
  --model gemma-3-12b \
  --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv \
  --output trans-en-ko

python3 finetune_trans.py --method merge  \
  --model gemma-3-12b \
  --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv \
  --output trans-en-ko

python3 test_trans.py --method inference \
  --model gemma-3-4b \
  --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv \
  --sentence "Uncle Henry never laughed. He worked hard from morning till night and did not know what joy was. He was gray also, from his long beard to his rough boots, and he looked stern and solemn, and rarely spoke."

python3 test_trans.py --method inference \
  --model gemma-3-12b \
  --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv \
  --sentence "Uncle Henry never laughed. He worked hard from morning till night and did not know what joy was. He was gray also, from his long beard to his rough boots, and he looked stern and solemn, and rarely spoke."
```

**gemma-3-4b**
vCPU 4, Memory 32GB<br>
finetune

```bash
{'train_runtime': 2862.3441, 'train_samples_per_second': 1.278, 'train_steps_per_second': 0.32, 'train_loss': 3.4040579561327324, 'mean_token_accuracy': 0.8790981863674364, 'epoch': 3.0}                                                              
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 915/915 [47:42<00:00,  3.13s/it]
Processing Files (5 / 5)                : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.88GB / 2.88GB,  154MB/s  
New Data Upload                         : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29.5MB / 29.5MB, 2.95MB/s  
  ...-3-4b-trans-en-ko/training_args.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.10kB / 6.10kB            
  ...l1-c.c.sayouzone-ai.internal.2138.0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33.1kB / 33.1kB            
  ...ma-3-4b-trans-en-ko/tokenizer.model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.69MB / 4.69MB            
  ...ans-en-ko/adapter_model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.84GB / 2.84GB            
  ...mma-3-4b-trans-en-ko/tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33.4MB / 33.4MB            
No files have been modified since last commit. Skipping to prevent empty commit.
model_name gemma-3-4b
output_dir gemma-3-4b-trans-en-ko
```

vCPU 4, Memory 32GB<br>
merge

```bash
model_name gemma-3-4b
output_dir gemma-3-4b-trans-en-ko
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:34<00:00, 17.15s/it]
```

vCPU 4, Memory 32GB<br>
inference

```bash
python3 test_trans.py --method inference \
  --model gemma-3-4b \
  --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv \
  --sentence "Uncle Henry never laughed. He worked hard from morning till night and did not know what joy was. He was gray also, from his long beard to his rough boots, and he looked stern and solemn, and rarely spoke."
```

```bash
args Namespace(method='inference', model='gemma-3-4b', csv_file='datasets/The_Wonderful_Wizard_of_Oz_1.csv', output='trans-en-ko', sentence='Uncle Henry never laughed. He worked hard from morning till night and did not know what joy was. He was gray also, from his long beard to his rough boots, and he looked stern and solemn, and rarely spoke.')
project sayouzone-ai
credentials <google.auth.compute_engine.credentials.Credentials object at 0x756b2a5e9f70>
credentials.service_account_email default
Dataset({
    features: ['genre', 'korean', 'english'],
    num_rows: 1219
})
model_name gemma-3-4b
output_dir gemma-3-4b-trans-en-ko
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:07<00:00,  6.77s/it]
inputs {'input_ids': tensor([[     2,  10354,  64017, 236787,    107,  40414,    506,   2269,   1816,
            699,   5422,    531,  21568,    618,  27646,  18355, 236761,    108,
          10354,  13065, 236787,    107, 168602,  12297,   2752,  39494, 236761,
           1293,   5934,   2651,    699,   5597,   8421,   3446,    532,   1602,
            711,   1281,   1144,  12690,    691, 236761,   1293,    691,  12819,
            992, 236764,    699,    914,   1440,  42603,    531,    914,  10887,
          22648, 236764,    532,    668,   6976,  47927,    532,  55962, 236764,
            532,  20390,  13804, 236761,    108,  10354,  14503, 236787,    107]],
       device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0')}
/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/_inductor/compile_fx.py:282: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
W0923 06:07:17.489000 1808 torch/_inductor/utils.py:1436] [0/0] Not enough SMs to use max_autotune_gemm mode
outputs tensor([[     2,  10354,  64017, 236787,    107,  40414,    506,   2269,   1816,
            699,   5422,    531,  21568,    618,  27646,  18355, 236761,    108,
          10354,  13065, 236787,    107, 168602,  12297,   2752,  39494, 236761,
           1293,   5934,   2651,    699,   5597,   8421,   3446,    532,   1602,
            711,   1281,   1144,  12690,    691, 236761,   1293,    691,  12819,
            992, 236764,    699,    914,   1440,  42603,    531,    914,  10887,
          22648, 236764,    532,    668,   6976,  47927,    532,  55962, 236764,
            532,  20390,  13804, 236761,    108,  10354,  14503, 236787,    107,
         239766, 237397, 246414, 237469,  73673, 181507, 114543, 113637, 243321,
         237131, 236761, 108997,   9182, 240876,  26691, 128093,  23971, 137587,
         237308,  88125, 221289,  67027, 236761,   9554, 193211, 111631, 237308,
          14108, 239234,  57206,  44628, 236761,   5815,  73673,  66635, 238608,
          26691,  14610, 242508, 237672, 215272,   9326, 240617,  23971,  38012,
         239870, 241488, 237293, 236743, 243801, 197701, 236764,  89435, 239379,
          83619,  26520, 237308,  26216,   9326, 239393,  41143, 123143, 155806,
         117024, 194984, 236761,      1]], device='cuda:0')
result ### Instruction:
Translate the following text from English to Korean as fantasy genre.

### Input:
Uncle Henry never laughed. He worked hard from morning till night and did not know what joy was. He was gray also, from his long beard to his rough boots, and he looked stern and solemn, and rarely spoke.

### Response:
언니헨리 또한 웃음을 몰랐다. 그는 아침부터 밤까지 쉬지 않고 일을 했다. 기분이 뭔지 모른 채였다. 그 또한 백발부터 거칠게 생긴 보츠까지 회색빛을 띠었고, 엄격하면서 진지하게 보였으며 거의 말을 하지 않았다.
response_part ['### Instruction:\nTranslate the following text from English to Korean as fantasy genre.\n\n### Input:\nUncle Henry never laughed. He worked hard from morning till night and did not know what joy was. He was gray also, from his long beard to his rough boots, and he looked stern and solemn, and rarely spoke.\n\n', '\n언니헨리 또한 웃음을 몰랐다. 그는 아침부터 밤까지 쉬지 않고 일을 했다. 기분이 뭔지 모른 채였다. 그 또한 백발부터 거칠게 생긴 보츠까지 회색빛을 띠었고, 엄격하면서 진지하게 보였으며 거의 말을 하지 않았다.']
outputs ['### Instruction:\nTranslate the following text from English to Korean as fantasy genre.\n\n### Input:\nUncle Henry never laughed. He worked hard from morning till night and did not know what joy was. He was gray also, from his long beard to his rough boots, and he looked stern and solemn, and rarely spoke.\n\n', '\n언니헨리 또한 웃음을 몰랐다. 그는 아침부터 밤까지 쉬지 않고 일을 했다. 기분이 뭔지 모른 채였다. 그 또한 백발부터 거칠게 생긴 보츠까지 회색빛을 띠었고, 엄격하면서 진지하게 보였으며 거의 말을 하지 않았다.']
```

**gemma-3-12b**
vCPU 4, Memory 32GB<br>
finetune

```bash
{'train_runtime': 6831.7155, 'train_samples_per_second': 0.535, 'train_steps_per_second': 0.134, 'train_loss': 2.9409341890303815, 'mean_token_accuracy': 0.9032163274915594, 'epoch': 3.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 915/915 [1:53:51<00:00,  7.47s/it]
Processing Files (5 / 5)                : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.36GB / 4.36GB,  154MB/s  
New Data Upload                         : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40.0MB / 40.0MB, 4.00MB/s  
  ...3-12b-trans-en-ko/training_args.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.10kB / 6.10kB            
  ...l1-c.c.sayouzone-ai.internal.4445.0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33.4kB / 33.4kB            
  ...a-3-12b-trans-en-ko/tokenizer.model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.69MB / 4.69MB            
  ...ma-3-12b-trans-en-ko/tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33.4MB / 33.4MB            
  ...ans-en-ko/adapter_model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.32GB / 4.32GB            
No files have been modified since last commit. Skipping to prevent empty commit.
model_name gemma-3-12b
output_dir gemma-3-12b-trans-en-ko
```

vCPU 16, Memory 64GB<br>
merge

```bash
model_name gemma-3-12b
output_dir gemma-3-12b-trans-en-ko
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:23<00:00, 16.67s/it]
```

##

**gemma-3-12b**
vCPU 4, Memory 32GB<br>
merge

```bash
model_name gemma-3-12b
output_dir gemma-3-12b-trans-en-ko
Loading checkpoint shards:  60%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                          | 3/5 [00:59<00:39, 19.66s/it]

```

vCPU 8, Memory 54GB<br>
merge

```bash
model_name gemma-3-12b
output_dir gemma-3-12b-trans-en-ko
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:23<00:00, 16.69s/it]
Killed
```

vCPU 4, Memory 32GB<br>
inference

```bash
model_name gemma-3-12b
output_dir gemma-3-12b-trans-en-ko
Loading checkpoint shards:  93%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 25/27 [01:20<00:06,  3.22s/it]
Traceback (most recent call last):
  File "/home/sjkim/gemma3/trans/test_trans.py", line 355, in <module>
    outputs = translate_genre(model_name, "fantasy", sentence)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3/trans/test_trans.py", line 264, in translate_genre
    model = AutoModelForCausalLM.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 600, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 317, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 5069, in from_pretrained
    ) = cls._load_pretrained_model(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 5532, in _load_pretrained_model
    _error_msgs, disk_offload_index, cpu_offload_index = load_shard_file(args)
                                                         ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 975, in load_shard_file
    disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 880, in _load_state_dict_into_meta_model
    _load_parameter_into_model(model, param_name, param.to(param_device))
                                                  ^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.75 GiB. GPU 0 has a total capacity of 22.05 GiB of which 3.17 GiB is free. Including non-PyTorch memory, this process has 18.86 GiB memory in use. Of the allocated memory 18.68 GiB is allocated by PyTorch, and 2.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

vCPU 8, Memory 54GB<br>
inference

```bash
model_name gemma-3-12b
output_dir gemma-3-12b-trans-en-ko
Loading checkpoint shards:  93%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 25/27 [01:20<00:06,  3.22s/it]
Traceback (most recent call last):
  File "/home/sjkim/gemma3/trans/test_trans.py", line 355, in <module>
    outputs = translate_genre(model_name, "fantasy", sentence)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3/trans/test_trans.py", line 264, in translate_genre
    model = AutoModelForCausalLM.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 600, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 317, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 5069, in from_pretrained
    ) = cls._load_pretrained_model(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 5532, in _load_pretrained_model
    _error_msgs, disk_offload_index, cpu_offload_index = load_shard_file(args)
                                                         ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 975, in load_shard_file
    disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 880, in _load_state_dict_into_meta_model
    _load_parameter_into_model(model, param_name, param.to(param_device))
                                                  ^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.75 GiB. GPU 0 has a total capacity of 22.05 GiB of which 3.17 GiB is free. Including non-PyTorch memory, this process has 18.86 GiB memory in use. Of the allocated memory 18.68 GiB is allocated by PyTorch, and 2.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

vCPU 16, Memory 64GB<br>
inference

```bash
args Namespace(method='inference', model='gemma-3-12b', csv_file='datasets/The_Wonderful_Wizard_of_Oz_1.csv', output='trans-en-ko', sentence='Uncle Henry never laughed. He worked hard from morning till night and did not know what joy was. He was gray also, from his long beard to his rough boots, and he looked stern and solemn, and rarely spoke.')
project sayouzone-ai
credentials <google.auth.compute_engine.credentials.Credentials object at 0x758d819a1670>
credentials.service_account_email default
Dataset({
    features: ['genre', 'korean', 'english'],
    num_rows: 1219
})
model_name gemma-3-12b
output_dir gemma-3-12b-trans-en-ko
Loading checkpoint shards:  93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋               | 25/27 [01:08<00:05,  2.75s/it]
Traceback (most recent call last):
  File "/home/sjkim/gemma3/trans/test_trans.py", line 355, in <module>
    outputs = translate_genre(model_name, "fantasy", sentence)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3/trans/test_trans.py", line 264, in translate_genre
    model = AutoModelForCausalLM.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 600, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 317, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 5069, in from_pretrained
    ) = cls._load_pretrained_model(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 5532, in _load_pretrained_model
    _error_msgs, disk_offload_index, cpu_offload_index = load_shard_file(args)
                                                         ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 975, in load_shard_file
    disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/sjkim/gemma3-env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 880, in _load_state_dict_into_meta_model
    _load_parameter_into_model(model, param_name, param.to(param_device))
                                                  ^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.75 GiB. GPU 0 has a total capacity of 22.05 GiB of which 3.17 GiB is free. Including non-PyTorch memory, this process has 18.86 GiB memory in use. Of the allocated memory 18.68 GiB is allocated by PyTorch, and 2.99 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```