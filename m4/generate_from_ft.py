from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
#from optimum.quanto import QuantoConfig
from transformers import QuantoConfig

import os
import torch

"""
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)

추가
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 베이스 모델 및 어댑터 경로 설정
#base_model_id = "google/gemma-3-4b-it"
#adapter_path = "./gemma3-ko-translation-adapter-final" # trainer.save_model()로 저장된 경로
base_model_id = "google/gemma-3-1b-it"
adapter_path = "./gemma-3-1b-it-en-ko-trans-final" # trainer.save_model()로 저장된 경로

# 2. 베이스 양자화 모델 로드
quantization_config = QuantoConfig(weights="int4")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    attn_implementation="eager",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# 3. PEFT 모델 로드 (베이스 모델에 어댑터 적용)
inference_model = PeftModel.from_pretrained(base_model, adapter_path)

# 4. 추론 함수 정의
def translate_text(model, text_to_translate):
    prompt = f"<s> 다음 영문 텍스트를 한국어로 번역하세요: {text_to_translate}"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    with torch.no_grad():
        # model.generate()는 2D 텐서를 반환합니다. (batch_size, sequence_length)
        outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    
    #response = tokenizer.decode(outputs, skip_special_tokens=True)
    # 응답에서 프롬프트 부분 제거
    #translated_text = response.split("")[-1].strip()
    
    # [수정점 1] 2D 텐서에서 1D 텐서를 추출합니다. (outputs[0])
    # [수정점 2] 원본 프롬프트를 제외하고, 새로 생성된 토큰들만 디코딩합니다.
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]

    translated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return translated_text

# 5. 비교 추론 실행
english_sentence = "Fine-tuning a large language model on Apple Silicon is now more accessible than ever."

print("--- 원본 Gemma 3 모델 추론 결과 ---")
original_translation = translate_text(base_model, english_sentence)
print(f"입력 (영어): {english_sentence}")
print(f"출력 (한국어): {original_translation}\n")


print("--- 미세조정된 모델 추론 결과 ---")
finetuned_translation = translate_text(inference_model, english_sentence)
print(f"입력 (영어): {english_sentence}")
print(f"출력 (한국어): {finetuned_translation}")

"""
Using device: mps
Using dtype: torch.bfloat16
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [00:00<00:00, 31102.26 examples/s]
SELECT Marketing_Campaigns.campaign_name FROM Marketing_Campaigns JOIN Movies ON Marketing_Campaigns.movie_id = Movies.movie_id WHERE Movies.release_year BETWEEN 2010 AND 2019 AND Marketing_Campaigns.marketing_budget <= 20000000;
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:453: UserWarning: Padding-free training is enabled, but the attention implementation is not set to 'flash_attention_2'. Padding-free training flattens batches into a single sequence, and 'flash_attention_2' is the only known attention mechanism that reliably supports this. Using other implementations may lead to unexpected behavior. To ensure compatibility, set `attn_implementation='flash_attention_2'` in the model configuration, or verify that your attention mechanism can handle flattened sequences.
  warnings.warn(
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:495: UserWarning: You are using packing, but the attention implementation is not set to 'flash_attention_2' or 'kernels-community/vllm-flash-attn3'. Packing flattens batches into a single sequence, and Flash Attention is the only known attention mechanisms that reliably support this. Using other implementations may lead to cross-contamination between batches. To avoid this, either disable packing by setting `packing=False`, or set `attn_implementation='flash_attention_2'` or `attn_implementation='kernels-community/vllm-flash-attn3'` in the model configuration.
  warnings.warn(
Tokenizing train dataset: 100%|███████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 5105.38 examples/s]
Packing train dataset: 100%|████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 165319.48 examples/s]
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 1.0136, 'grad_norm': 1.236979603767395, 'learning_rate': 0.0002, 'num_tokens': 40026.0, 'mean_token_accuracy': 0.8128562897443772, 'epoch': 0.02}
{'loss': 0.5957, 'grad_norm': 0.9213776588439941, 'learning_rate': 0.0002, 'num_tokens': 79834.0, 'mean_token_accuracy': 0.8547919601202011, 'epoch': 0.04}
{'loss': 0.5338, 'grad_norm': 0.7811344861984253, 'learning_rate': 0.0002, 'num_tokens': 119323.0, 'mean_token_accuracy': 0.8677195474505425, 'epoch': 0.06}
{'loss': 0.5039, 'grad_norm': 0.7371190786361694, 'learning_rate': 0.0002, 'num_tokens': 159671.0, 'mean_token_accuracy': 0.868143393099308, 'epoch': 0.07}
{'loss': 0.5038, 'grad_norm': 0.7669692635536194, 'learning_rate': 0.0002, 'num_tokens': 199497.0, 'mean_token_accuracy': 0.8729725852608681, 'epoch': 0.09}
{'loss': 0.4864, 'grad_norm': 0.6642109751701355, 'learning_rate': 0.0002, 'num_tokens': 239482.0, 'mean_token_accuracy': 0.8767930209636688, 'epoch': 0.11}
{'loss': 0.4563, 'grad_norm': 0.6802425384521484, 'learning_rate': 0.0002, 'num_tokens': 279534.0, 'mean_token_accuracy': 0.881158997118473, 'epoch': 0.13}
{'loss': 0.4638, 'grad_norm': 0.687825620174408, 'learning_rate': 0.0002, 'num_tokens': 319202.0, 'mean_token_accuracy': 0.8762027680873871, 'epoch': 0.15}
{'loss': 0.4693, 'grad_norm': 0.7378871440887451, 'learning_rate': 0.0002, 'num_tokens': 359163.0, 'mean_token_accuracy': 0.8775609970092774, 'epoch': 0.17}
{'loss': 0.4483, 'grad_norm': 0.7447844743728638, 'learning_rate': 0.0002, 'num_tokens': 398928.0, 'mean_token_accuracy': 0.8813143759965897, 'epoch': 0.19}
{'loss': 0.4704, 'grad_norm': 0.6233909130096436, 'learning_rate': 0.0002, 'num_tokens': 439166.0, 'mean_token_accuracy': 0.8785416021943092, 'epoch': 0.2}
{'loss': 0.4681, 'grad_norm': 0.6493863463401794, 'learning_rate': 0.0002, 'num_tokens': 478863.0, 'mean_token_accuracy': 0.8810286238789559, 'epoch': 0.22}
{'loss': 0.4588, 'grad_norm': 0.6366643309593201, 'learning_rate': 0.0002, 'num_tokens': 518591.0, 'mean_token_accuracy': 0.8808981344103813, 'epoch': 0.24}
{'loss': 0.4606, 'grad_norm': 0.6141524910926819, 'learning_rate': 0.0002, 'num_tokens': 558346.0, 'mean_token_accuracy': 0.880066342651844, 'epoch': 0.26}
{'loss': 0.4528, 'grad_norm': 0.7220727801322937, 'learning_rate': 0.0002, 'num_tokens': 598330.0, 'mean_token_accuracy': 0.8799679234623909, 'epoch': 0.28}
{'loss': 0.4312, 'grad_norm': 0.5953671336174011, 'learning_rate': 0.0002, 'num_tokens': 638262.0, 'mean_token_accuracy': 0.8862154319882393, 'epoch': 0.3}
{'loss': 0.4591, 'grad_norm': 0.5965115427970886, 'learning_rate': 0.0002, 'num_tokens': 678310.0, 'mean_token_accuracy': 0.8806457713246345, 'epoch': 0.32}
{'loss': 0.4389, 'grad_norm': 0.6372408270835876, 'learning_rate': 0.0002, 'num_tokens': 718528.0, 'mean_token_accuracy': 0.8857690691947937, 'epoch': 0.33}
{'loss': 0.4515, 'grad_norm': 0.6634373068809509, 'learning_rate': 0.0002, 'num_tokens': 758617.0, 'mean_token_accuracy': 0.8825202658772469, 'epoch': 0.35}
{'loss': 0.4499, 'grad_norm': 0.6056446433067322, 'learning_rate': 0.0002, 'num_tokens': 798637.0, 'mean_token_accuracy': 0.883007661998272, 'epoch': 0.37}
{'loss': 0.4265, 'grad_norm': 0.6697975993156433, 'learning_rate': 0.0002, 'num_tokens': 838972.0, 'mean_token_accuracy': 0.8855334594845772, 'epoch': 0.39}
{'loss': 0.4277, 'grad_norm': 0.6101330518722534, 'learning_rate': 0.0002, 'num_tokens': 878351.0, 'mean_token_accuracy': 0.887157553434372, 'epoch': 0.41}
{'loss': 0.4367, 'grad_norm': 0.5993234515190125, 'learning_rate': 0.0002, 'num_tokens': 918373.0, 'mean_token_accuracy': 0.8829835429787636, 'epoch': 0.43}
{'loss': 0.4231, 'grad_norm': 0.7631500959396362, 'learning_rate': 0.0002, 'num_tokens': 958537.0, 'mean_token_accuracy': 0.8884492322802544, 'epoch': 0.45}
{'loss': 0.4313, 'grad_norm': 0.639520525932312, 'learning_rate': 0.0002, 'num_tokens': 998714.0, 'mean_token_accuracy': 0.8873802915215492, 'epoch': 0.46}
{'loss': 0.4353, 'grad_norm': 0.625914454460144, 'learning_rate': 0.0002, 'num_tokens': 1038593.0, 'mean_token_accuracy': 0.8846580937504769, 'epoch': 0.48}
{'loss': 0.4268, 'grad_norm': 0.6579174995422363, 'learning_rate': 0.0002, 'num_tokens': 1078130.0, 'mean_token_accuracy': 0.8871070474386216, 'epoch': 0.5}
{'loss': 0.4176, 'grad_norm': 0.6097962856292725, 'learning_rate': 0.0002, 'num_tokens': 1118109.0, 'mean_token_accuracy': 0.8889595285058022, 'epoch': 0.52}
{'loss': 0.4326, 'grad_norm': 0.658940315246582, 'learning_rate': 0.0002, 'num_tokens': 1158128.0, 'mean_token_accuracy': 0.8855308324098587, 'epoch': 0.54}
{'loss': 0.4316, 'grad_norm': 0.6073350310325623, 'learning_rate': 0.0002, 'num_tokens': 1197955.0, 'mean_token_accuracy': 0.8847613856196404, 'epoch': 0.56}
{'loss': 0.4067, 'grad_norm': 0.6056622862815857, 'learning_rate': 0.0002, 'num_tokens': 1237179.0, 'mean_token_accuracy': 0.8911844044923782, 'epoch': 0.58}
{'loss': 0.4383, 'grad_norm': 0.672029972076416, 'learning_rate': 0.0002, 'num_tokens': 1277098.0, 'mean_token_accuracy': 0.8853227883577347, 'epoch': 0.6}
{'loss': 0.4207, 'grad_norm': 0.5153404474258423, 'learning_rate': 0.0002, 'num_tokens': 1316867.0, 'mean_token_accuracy': 0.8878092855215073, 'epoch': 0.61}
{'loss': 0.4246, 'grad_norm': 0.5592024922370911, 'learning_rate': 0.0002, 'num_tokens': 1356413.0, 'mean_token_accuracy': 0.8864561468362808, 'epoch': 0.63}
{'loss': 0.4246, 'grad_norm': 0.5504661202430725, 'learning_rate': 0.0002, 'num_tokens': 1397032.0, 'mean_token_accuracy': 0.8882461801171303, 'epoch': 0.65}
{'loss': 0.4008, 'grad_norm': 0.5946975350379944, 'learning_rate': 0.0002, 'num_tokens': 1437067.0, 'mean_token_accuracy': 0.8931970775127411, 'epoch': 0.67}
{'loss': 0.4304, 'grad_norm': 0.6204308271408081, 'learning_rate': 0.0002, 'num_tokens': 1477113.0, 'mean_token_accuracy': 0.8860071673989296, 'epoch': 0.69}
{'loss': 0.3908, 'grad_norm': 0.5584782361984253, 'learning_rate': 0.0002, 'num_tokens': 1517126.0, 'mean_token_accuracy': 0.8944241181015968, 'epoch': 0.71}
{'loss': 0.4099, 'grad_norm': 0.5410889983177185, 'learning_rate': 0.0002, 'num_tokens': 1556834.0, 'mean_token_accuracy': 0.8915607750415802, 'epoch': 0.73}
{'loss': 0.4123, 'grad_norm': 0.5343829989433289, 'learning_rate': 0.0002, 'num_tokens': 1596813.0, 'mean_token_accuracy': 0.8883616089820862, 'epoch': 0.74}
{'loss': 0.4164, 'grad_norm': 0.6330368518829346, 'learning_rate': 0.0002, 'num_tokens': 1637029.0, 'mean_token_accuracy': 0.8874604493379593, 'epoch': 0.76}
{'loss': 0.4219, 'grad_norm': 0.6169411540031433, 'learning_rate': 0.0002, 'num_tokens': 1677213.0, 'mean_token_accuracy': 0.8882613807916642, 'epoch': 0.78}
{'loss': 0.4071, 'grad_norm': 0.5435759425163269, 'learning_rate': 0.0002, 'num_tokens': 1717048.0, 'mean_token_accuracy': 0.8894449457526207, 'epoch': 0.8}
{'loss': 0.4146, 'grad_norm': 0.5536595582962036, 'learning_rate': 0.0002, 'num_tokens': 1756946.0, 'mean_token_accuracy': 0.889171588420868, 'epoch': 0.82}
{'loss': 0.4125, 'grad_norm': 0.5437877178192139, 'learning_rate': 0.0002, 'num_tokens': 1796877.0, 'mean_token_accuracy': 0.8902988284826279, 'epoch': 0.84}
{'loss': 0.4124, 'grad_norm': 0.5602271556854248, 'learning_rate': 0.0002, 'num_tokens': 1837030.0, 'mean_token_accuracy': 0.8905262485146522, 'epoch': 0.86}
{'loss': 0.4093, 'grad_norm': 0.5430951714515686, 'learning_rate': 0.0002, 'num_tokens': 1876800.0, 'mean_token_accuracy': 0.8907581895589829, 'epoch': 0.87}
{'loss': 0.4204, 'grad_norm': 0.5204606056213379, 'learning_rate': 0.0002, 'num_tokens': 1916699.0, 'mean_token_accuracy': 0.8858507335186004, 'epoch': 0.89}
{'loss': 0.4117, 'grad_norm': 0.644989550113678, 'learning_rate': 0.0002, 'num_tokens': 1956319.0, 'mean_token_accuracy': 0.8896844938397408, 'epoch': 0.91}
{'loss': 0.4019, 'grad_norm': 0.5185262560844421, 'learning_rate': 0.0002, 'num_tokens': 1996295.0, 'mean_token_accuracy': 0.8907623708248138, 'epoch': 0.93}
{'loss': 0.4017, 'grad_norm': 0.5461589097976685, 'learning_rate': 0.0002, 'num_tokens': 2035176.0, 'mean_token_accuracy': 0.8930577024817467, 'epoch': 0.95}
{'loss': 0.3846, 'grad_norm': 0.5392025709152222, 'learning_rate': 0.0002, 'num_tokens': 2075335.0, 'mean_token_accuracy': 0.8940192595124244, 'epoch': 0.97}
{'loss': 0.4092, 'grad_norm': 0.5368521213531494, 'learning_rate': 0.0002, 'num_tokens': 2115017.0, 'mean_token_accuracy': 0.8907889097929, 'epoch': 0.99}
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.3782, 'grad_norm': 0.4973786771297455, 'learning_rate': 0.0002, 'num_tokens': 2153729.0, 'mean_token_accuracy': 0.8953530757855146, 'epoch': 1.0}
{'loss': 0.3391, 'grad_norm': 0.4807419180870056, 'learning_rate': 0.0002, 'num_tokens': 2193360.0, 'mean_token_accuracy': 0.9027973547577858, 'epoch': 1.02}
{'loss': 0.3171, 'grad_norm': 0.555351734161377, 'learning_rate': 0.0002, 'num_tokens': 2233024.0, 'mean_token_accuracy': 0.9082618728280067, 'epoch': 1.04}
{'loss': 0.32, 'grad_norm': 0.5669279098510742, 'learning_rate': 0.0002, 'num_tokens': 2273300.0, 'mean_token_accuracy': 0.9073583871126175, 'epoch': 1.06}
{'loss': 0.3248, 'grad_norm': 0.5113018751144409, 'learning_rate': 0.0002, 'num_tokens': 2313286.0, 'mean_token_accuracy': 0.9053036078810692, 'epoch': 1.08}
{'loss': 0.3227, 'grad_norm': 0.5345684885978699, 'learning_rate': 0.0002, 'num_tokens': 2353531.0, 'mean_token_accuracy': 0.9071148276329041, 'epoch': 1.1}
{'loss': 0.3222, 'grad_norm': 0.5311261415481567, 'learning_rate': 0.0002, 'num_tokens': 2393520.0, 'mean_token_accuracy': 0.9065080195665359, 'epoch': 1.12}
{'loss': 0.3224, 'grad_norm': 0.5430824756622314, 'learning_rate': 0.0002, 'num_tokens': 2433525.0, 'mean_token_accuracy': 0.9060756713151932, 'epoch': 1.13}
{'loss': 0.321, 'grad_norm': 0.5516654849052429, 'learning_rate': 0.0002, 'num_tokens': 2473277.0, 'mean_token_accuracy': 0.9056764379143715, 'epoch': 1.15}
{'loss': 0.3242, 'grad_norm': 0.5204035639762878, 'learning_rate': 0.0002, 'num_tokens': 2513307.0, 'mean_token_accuracy': 0.9058673143386841, 'epoch': 1.17}
{'loss': 0.3153, 'grad_norm': 0.5102381110191345, 'learning_rate': 0.0002, 'num_tokens': 2553498.0, 'mean_token_accuracy': 0.9082729563117027, 'epoch': 1.19}
{'loss': 0.3294, 'grad_norm': 0.557854950428009, 'learning_rate': 0.0002, 'num_tokens': 2593371.0, 'mean_token_accuracy': 0.9032549217343331, 'epoch': 1.21}
{'loss': 0.3178, 'grad_norm': 0.508311927318573, 'learning_rate': 0.0002, 'num_tokens': 2633327.0, 'mean_token_accuracy': 0.9090947896242142, 'epoch': 1.23}
{'loss': 0.3208, 'grad_norm': 0.5488402843475342, 'learning_rate': 0.0002, 'num_tokens': 2673280.0, 'mean_token_accuracy': 0.9062734305858612, 'epoch': 1.25}
{'loss': 0.3323, 'grad_norm': 0.5938761234283447, 'learning_rate': 0.0002, 'num_tokens': 2713370.0, 'mean_token_accuracy': 0.9041984036564827, 'epoch': 1.26}
{'loss': 0.3232, 'grad_norm': 0.5648936629295349, 'learning_rate': 0.0002, 'num_tokens': 2753325.0, 'mean_token_accuracy': 0.9053479865193367, 'epoch': 1.28}
{'loss': 0.3264, 'grad_norm': 0.5131396651268005, 'learning_rate': 0.0002, 'num_tokens': 2793292.0, 'mean_token_accuracy': 0.9045232102274895, 'epoch': 1.3}
{'loss': 0.3213, 'grad_norm': 0.4814319610595703, 'learning_rate': 0.0002, 'num_tokens': 2832731.0, 'mean_token_accuracy': 0.9088704749941826, 'epoch': 1.32}
{'loss': 0.3315, 'grad_norm': 0.545452356338501, 'learning_rate': 0.0002, 'num_tokens': 2872928.0, 'mean_token_accuracy': 0.9046394735574722, 'epoch': 1.34}
{'loss': 0.3243, 'grad_norm': 0.5669392347335815, 'learning_rate': 0.0002, 'num_tokens': 2912851.0, 'mean_token_accuracy': 0.9080562934279441, 'epoch': 1.36}
{'loss': 0.3292, 'grad_norm': 0.5475120544433594, 'learning_rate': 0.0002, 'num_tokens': 2952170.0, 'mean_token_accuracy': 0.9038760870695114, 'epoch': 1.38}
{'loss': 0.323, 'grad_norm': 0.539612352848053, 'learning_rate': 0.0002, 'num_tokens': 2992139.0, 'mean_token_accuracy': 0.9054552987217903, 'epoch': 1.39}
{'loss': 0.3474, 'grad_norm': 0.5626351833343506, 'learning_rate': 0.0002, 'num_tokens': 3032023.0, 'mean_token_accuracy': 0.8997958049178123, 'epoch': 1.41}
{'loss': 0.3199, 'grad_norm': 0.5492633581161499, 'learning_rate': 0.0002, 'num_tokens': 3071813.0, 'mean_token_accuracy': 0.9060791879892349, 'epoch': 1.43}
{'loss': 0.3198, 'grad_norm': 0.5483739376068115, 'learning_rate': 0.0002, 'num_tokens': 3111518.0, 'mean_token_accuracy': 0.9070928260684014, 'epoch': 1.45}
{'loss': 0.3251, 'grad_norm': 0.5675632953643799, 'learning_rate': 0.0002, 'num_tokens': 3151178.0, 'mean_token_accuracy': 0.9069820284843445, 'epoch': 1.47}
{'loss': 0.3178, 'grad_norm': 0.4787401556968689, 'learning_rate': 0.0002, 'num_tokens': 3191294.0, 'mean_token_accuracy': 0.9097749590873718, 'epoch': 1.49}
{'loss': 0.3309, 'grad_norm': 0.5823889374732971, 'learning_rate': 0.0002, 'num_tokens': 3231162.0, 'mean_token_accuracy': 0.9057106748223305, 'epoch': 1.51}
{'loss': 0.3218, 'grad_norm': 0.5110133290290833, 'learning_rate': 0.0002, 'num_tokens': 3271251.0, 'mean_token_accuracy': 0.9070012211799622, 'epoch': 1.52}
{'loss': 0.3206, 'grad_norm': 0.5461782217025757, 'learning_rate': 0.0002, 'num_tokens': 3310929.0, 'mean_token_accuracy': 0.9080902665853501, 'epoch': 1.54}
{'loss': 0.3328, 'grad_norm': 0.5387473106384277, 'learning_rate': 0.0002, 'num_tokens': 3351097.0, 'mean_token_accuracy': 0.9056071519851685, 'epoch': 1.56}
{'loss': 0.3292, 'grad_norm': 0.5289240479469299, 'learning_rate': 0.0002, 'num_tokens': 3391312.0, 'mean_token_accuracy': 0.9041397139430046, 'epoch': 1.58}
{'loss': 0.3353, 'grad_norm': 0.5780121088027954, 'learning_rate': 0.0002, 'num_tokens': 3431161.0, 'mean_token_accuracy': 0.9028048485517501, 'epoch': 1.6}
{'loss': 0.3386, 'grad_norm': 0.5762632489204407, 'learning_rate': 0.0002, 'num_tokens': 3471016.0, 'mean_token_accuracy': 0.9028788968920708, 'epoch': 1.62}
{'loss': 0.3301, 'grad_norm': 0.5224876403808594, 'learning_rate': 0.0002, 'num_tokens': 3511157.0, 'mean_token_accuracy': 0.9042042046785355, 'epoch': 1.64}
{'loss': 0.3252, 'grad_norm': 0.5193665623664856, 'learning_rate': 0.0002, 'num_tokens': 3550527.0, 'mean_token_accuracy': 0.9081370294094085, 'epoch': 1.65}
{'loss': 0.3258, 'grad_norm': 0.513916552066803, 'learning_rate': 0.0002, 'num_tokens': 3590132.0, 'mean_token_accuracy': 0.9067825883626938, 'epoch': 1.67}
{'loss': 0.3296, 'grad_norm': 0.5208548307418823, 'learning_rate': 0.0002, 'num_tokens': 3629754.0, 'mean_token_accuracy': 0.905221575498581, 'epoch': 1.69}
{'loss': 0.3315, 'grad_norm': 0.5431376695632935, 'learning_rate': 0.0002, 'num_tokens': 3669628.0, 'mean_token_accuracy': 0.9051292940974236, 'epoch': 1.71}
{'loss': 0.3371, 'grad_norm': 0.7013857364654541, 'learning_rate': 0.0002, 'num_tokens': 3709191.0, 'mean_token_accuracy': 0.9017755821347236, 'epoch': 1.73}
{'loss': 0.3273, 'grad_norm': 0.5057289600372314, 'learning_rate': 0.0002, 'num_tokens': 3748885.0, 'mean_token_accuracy': 0.9064222559332847, 'epoch': 1.75}
{'loss': 0.3188, 'grad_norm': 0.5481709241867065, 'learning_rate': 0.0002, 'num_tokens': 3789129.0, 'mean_token_accuracy': 0.9079301953315735, 'epoch': 1.77}
{'loss': 0.3253, 'grad_norm': 0.6183758974075317, 'learning_rate': 0.0002, 'num_tokens': 3829107.0, 'mean_token_accuracy': 0.9073418602347374, 'epoch': 1.78}
{'loss': 0.3258, 'grad_norm': 0.5211597084999084, 'learning_rate': 0.0002, 'num_tokens': 3869457.0, 'mean_token_accuracy': 0.9055126816034317, 'epoch': 1.8}
{'loss': 0.325, 'grad_norm': 0.5190044641494751, 'learning_rate': 0.0002, 'num_tokens': 3909213.0, 'mean_token_accuracy': 0.9046908274292946, 'epoch': 1.82}
{'loss': 0.3322, 'grad_norm': 0.5414463877677917, 'learning_rate': 0.0002, 'num_tokens': 3949215.0, 'mean_token_accuracy': 0.9045024946331978, 'epoch': 1.84}
{'loss': 0.335, 'grad_norm': 0.536609947681427, 'learning_rate': 0.0002, 'num_tokens': 3989029.0, 'mean_token_accuracy': 0.9051804319024086, 'epoch': 1.86}
{'loss': 0.3236, 'grad_norm': 0.49994465708732605, 'learning_rate': 0.0002, 'num_tokens': 4029002.0, 'mean_token_accuracy': 0.9049234464764595, 'epoch': 1.88}
{'loss': 0.3374, 'grad_norm': 0.5066275000572205, 'learning_rate': 0.0002, 'num_tokens': 4068702.0, 'mean_token_accuracy': 0.9028499156236649, 'epoch': 1.9}
{'loss': 0.3216, 'grad_norm': 0.5861412882804871, 'learning_rate': 0.0002, 'num_tokens': 4108820.0, 'mean_token_accuracy': 0.9059081062674522, 'epoch': 1.91}
{'loss': 0.331, 'grad_norm': 0.5454859733581543, 'learning_rate': 0.0002, 'num_tokens': 4149073.0, 'mean_token_accuracy': 0.9026151299476624, 'epoch': 1.93}
{'loss': 0.3282, 'grad_norm': 0.531156599521637, 'learning_rate': 0.0002, 'num_tokens': 4188837.0, 'mean_token_accuracy': 0.9043647781014442, 'epoch': 1.95}
{'loss': 0.3219, 'grad_norm': 0.526774525642395, 'learning_rate': 0.0002, 'num_tokens': 4228750.0, 'mean_token_accuracy': 0.907652473449707, 'epoch': 1.97}
{'loss': 0.3359, 'grad_norm': 0.47987040877342224, 'learning_rate': 0.0002, 'num_tokens': 4268700.0, 'mean_token_accuracy': 0.9039060860872269, 'epoch': 1.99}
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.3032, 'grad_norm': 0.46186354756355286, 'learning_rate': 0.0002, 'num_tokens': 4307018.0, 'mean_token_accuracy': 0.913516579530178, 'epoch': 2.01}
{'loss': 0.2464, 'grad_norm': 0.5586009621620178, 'learning_rate': 0.0002, 'num_tokens': 4346422.0, 'mean_token_accuracy': 0.9267527103424072, 'epoch': 2.03}
{'loss': 0.2481, 'grad_norm': 0.5081720352172852, 'learning_rate': 0.0002, 'num_tokens': 4386753.0, 'mean_token_accuracy': 0.9253434732556343, 'epoch': 2.04}
{'loss': 0.2577, 'grad_norm': 0.5094988942146301, 'learning_rate': 0.0002, 'num_tokens': 4426715.0, 'mean_token_accuracy': 0.9232575505971908, 'epoch': 2.06}
{'loss': 0.256, 'grad_norm': 0.5542491674423218, 'learning_rate': 0.0002, 'num_tokens': 4466487.0, 'mean_token_accuracy': 0.9222521096467972, 'epoch': 2.08}
{'loss': 0.259, 'grad_norm': 0.5145403742790222, 'learning_rate': 0.0002, 'num_tokens': 4506285.0, 'mean_token_accuracy': 0.9212210968136787, 'epoch': 2.1}
{'loss': 0.2513, 'grad_norm': 0.5744456648826599, 'learning_rate': 0.0002, 'num_tokens': 4545934.0, 'mean_token_accuracy': 0.9226992711424827, 'epoch': 2.12}
{'loss': 0.2577, 'grad_norm': 0.5146250128746033, 'learning_rate': 0.0002, 'num_tokens': 4586146.0, 'mean_token_accuracy': 0.9208047106862068, 'epoch': 2.14}
{'loss': 0.2582, 'grad_norm': 0.5610396862030029, 'learning_rate': 0.0002, 'num_tokens': 4626164.0, 'mean_token_accuracy': 0.9209060207009315, 'epoch': 2.16}
{'loss': 0.2585, 'grad_norm': 0.5822878479957581, 'learning_rate': 0.0002, 'num_tokens': 4666034.0, 'mean_token_accuracy': 0.9226884454488754, 'epoch': 2.17}
{'loss': 0.265, 'grad_norm': 0.527222752571106, 'learning_rate': 0.0002, 'num_tokens': 4705465.0, 'mean_token_accuracy': 0.9183770149946213, 'epoch': 2.19}
{'loss': 0.2564, 'grad_norm': 0.5210466384887695, 'learning_rate': 0.0002, 'num_tokens': 4745637.0, 'mean_token_accuracy': 0.9233832776546478, 'epoch': 2.21}
{'loss': 0.2629, 'grad_norm': 0.5471710562705994, 'learning_rate': 0.0002, 'num_tokens': 4785580.0, 'mean_token_accuracy': 0.9207369282841682, 'epoch': 2.23}
{'loss': 0.2612, 'grad_norm': 0.6638558506965637, 'learning_rate': 0.0002, 'num_tokens': 4825473.0, 'mean_token_accuracy': 0.9205511644482612, 'epoch': 2.25}
{'loss': 0.2684, 'grad_norm': 0.5229365229606628, 'learning_rate': 0.0002, 'num_tokens': 4865562.0, 'mean_token_accuracy': 0.9198095753788949, 'epoch': 2.27}
{'loss': 0.2623, 'grad_norm': 0.5354583859443665, 'learning_rate': 0.0002, 'num_tokens': 4905533.0, 'mean_token_accuracy': 0.9215421035885811, 'epoch': 2.29}
{'loss': 0.2647, 'grad_norm': 0.5281250476837158, 'learning_rate': 0.0002, 'num_tokens': 4945638.0, 'mean_token_accuracy': 0.9218300610780716, 'epoch': 2.3}
{'loss': 0.2651, 'grad_norm': 0.5621822476387024, 'learning_rate': 0.0002, 'num_tokens': 4984804.0, 'mean_token_accuracy': 0.9197034910321236, 'epoch': 2.32}
{'loss': 0.2646, 'grad_norm': 0.8499956727027893, 'learning_rate': 0.0002, 'num_tokens': 5024798.0, 'mean_token_accuracy': 0.9182656943798065, 'epoch': 2.34}
{'loss': 0.2588, 'grad_norm': 0.5596104860305786, 'learning_rate': 0.0002, 'num_tokens': 5064757.0, 'mean_token_accuracy': 0.9227040186524391, 'epoch': 2.36}
{'loss': 0.2642, 'grad_norm': 0.568699300289154, 'learning_rate': 0.0002, 'num_tokens': 5104885.0, 'mean_token_accuracy': 0.9201527044177056, 'epoch': 2.38}
{'loss': 0.2629, 'grad_norm': 0.562370777130127, 'learning_rate': 0.0002, 'num_tokens': 5144596.0, 'mean_token_accuracy': 0.9194921344518662, 'epoch': 2.4}
{'loss': 0.2598, 'grad_norm': 0.5325804352760315, 'learning_rate': 0.0002, 'num_tokens': 5184777.0, 'mean_token_accuracy': 0.9211806029081344, 'epoch': 2.42}
{'loss': 0.2551, 'grad_norm': 0.575689435005188, 'learning_rate': 0.0002, 'num_tokens': 5224611.0, 'mean_token_accuracy': 0.9232968211174011, 'epoch': 2.44}
{'loss': 0.2624, 'grad_norm': 0.5146995186805725, 'learning_rate': 0.0002, 'num_tokens': 5264675.0, 'mean_token_accuracy': 0.9211074694991112, 'epoch': 2.45}
{'loss': 0.2688, 'grad_norm': 0.5256116390228271, 'learning_rate': 0.0002, 'num_tokens': 5304615.0, 'mean_token_accuracy': 0.9195909634232521, 'epoch': 2.47}
{'loss': 0.2712, 'grad_norm': 0.496093213558197, 'learning_rate': 0.0002, 'num_tokens': 5344885.0, 'mean_token_accuracy': 0.9203537598252296, 'epoch': 2.49}
{'loss': 0.2606, 'grad_norm': 0.5949134230613708, 'learning_rate': 0.0002, 'num_tokens': 5384952.0, 'mean_token_accuracy': 0.9215462356805801, 'epoch': 2.51}
{'loss': 0.2617, 'grad_norm': 0.545548141002655, 'learning_rate': 0.0002, 'num_tokens': 5424855.0, 'mean_token_accuracy': 0.9222137272357941, 'epoch': 2.53}
{'loss': 0.2703, 'grad_norm': 0.5472376346588135, 'learning_rate': 0.0002, 'num_tokens': 5464819.0, 'mean_token_accuracy': 0.919766204059124, 'epoch': 2.55}
{'loss': 0.2648, 'grad_norm': 0.5530557632446289, 'learning_rate': 0.0002, 'num_tokens': 5504940.0, 'mean_token_accuracy': 0.9208953931927681, 'epoch': 2.57}
{'loss': 0.2611, 'grad_norm': 0.5395246744155884, 'learning_rate': 0.0002, 'num_tokens': 5545206.0, 'mean_token_accuracy': 0.921547019481659, 'epoch': 2.58}
{'loss': 0.272, 'grad_norm': 0.5963030457496643, 'learning_rate': 0.0002, 'num_tokens': 5585270.0, 'mean_token_accuracy': 0.9165272042155266, 'epoch': 2.6}
{'loss': 0.2699, 'grad_norm': 0.5615944862365723, 'learning_rate': 0.0002, 'num_tokens': 5624823.0, 'mean_token_accuracy': 0.9178289130330086, 'epoch': 2.62}
{'loss': 0.2655, 'grad_norm': 0.591606616973877, 'learning_rate': 0.0002, 'num_tokens': 5664724.0, 'mean_token_accuracy': 0.9191545233130455, 'epoch': 2.64}
{'loss': 0.2622, 'grad_norm': 0.5506940484046936, 'learning_rate': 0.0002, 'num_tokens': 5704707.0, 'mean_token_accuracy': 0.920456363260746, 'epoch': 2.66}
{'loss': 0.2648, 'grad_norm': 0.5572147965431213, 'learning_rate': 0.0002, 'num_tokens': 5744687.0, 'mean_token_accuracy': 0.9205353111028671, 'epoch': 2.68}
{'loss': 0.2669, 'grad_norm': 0.5384182929992676, 'learning_rate': 0.0002, 'num_tokens': 5784581.0, 'mean_token_accuracy': 0.9189376145601272, 'epoch': 2.7}
{'loss': 0.2723, 'grad_norm': 0.6729273796081543, 'learning_rate': 0.0002, 'num_tokens': 5824784.0, 'mean_token_accuracy': 0.9156325370073318, 'epoch': 2.71}
{'loss': 0.2741, 'grad_norm': 0.5370689630508423, 'learning_rate': 0.0002, 'num_tokens': 5864989.0, 'mean_token_accuracy': 0.9169686585664749, 'epoch': 2.73}
{'loss': 0.2704, 'grad_norm': 0.5548659563064575, 'learning_rate': 0.0002, 'num_tokens': 5904824.0, 'mean_token_accuracy': 0.9195882186293602, 'epoch': 2.75}
{'loss': 0.2649, 'grad_norm': 0.5700987577438354, 'learning_rate': 0.0002, 'num_tokens': 5943983.0, 'mean_token_accuracy': 0.9200750604271889, 'epoch': 2.77}
{'loss': 0.2652, 'grad_norm': 0.5408449769020081, 'learning_rate': 0.0002, 'num_tokens': 5983598.0, 'mean_token_accuracy': 0.9194610670208931, 'epoch': 2.79}
{'loss': 0.2607, 'grad_norm': 0.5261676907539368, 'learning_rate': 0.0002, 'num_tokens': 6023417.0, 'mean_token_accuracy': 0.9222517400979996, 'epoch': 2.81}
{'loss': 0.2694, 'grad_norm': 0.48337745666503906, 'learning_rate': 0.0002, 'num_tokens': 6063426.0, 'mean_token_accuracy': 0.9215697407722473, 'epoch': 2.83}
{'loss': 0.2673, 'grad_norm': 0.5304609537124634, 'learning_rate': 0.0002, 'num_tokens': 6102623.0, 'mean_token_accuracy': 0.918236993253231, 'epoch': 2.84}
{'loss': 0.2702, 'grad_norm': 0.5223886966705322, 'learning_rate': 0.0002, 'num_tokens': 6142351.0, 'mean_token_accuracy': 0.919144082069397, 'epoch': 2.86}
{'loss': 0.2687, 'grad_norm': 0.5346062779426575, 'learning_rate': 0.0002, 'num_tokens': 6182022.0, 'mean_token_accuracy': 0.9200467303395271, 'epoch': 2.88}
{'loss': 0.268, 'grad_norm': 0.5096137523651123, 'learning_rate': 0.0002, 'num_tokens': 6221714.0, 'mean_token_accuracy': 0.9205083504319191, 'epoch': 2.9}
{'loss': 0.2693, 'grad_norm': 0.5935035943984985, 'learning_rate': 0.0002, 'num_tokens': 6261689.0, 'mean_token_accuracy': 0.9190357699990273, 'epoch': 2.92}
{'loss': 0.2711, 'grad_norm': 0.6035186052322388, 'learning_rate': 0.0002, 'num_tokens': 6301779.0, 'mean_token_accuracy': 0.9179719686508179, 'epoch': 2.94}
{'loss': 0.271, 'grad_norm': 0.5242613554000854, 'learning_rate': 0.0002, 'num_tokens': 6341985.0, 'mean_token_accuracy': 0.9188696756958962, 'epoch': 2.96}
{'loss': 0.2745, 'grad_norm': 0.5373081564903259, 'learning_rate': 0.0002, 'num_tokens': 6381917.0, 'mean_token_accuracy': 0.9182656198740006, 'epoch': 2.97}
{'loss': 0.2705, 'grad_norm': 0.5248370170593262, 'learning_rate': 0.0002, 'num_tokens': 6421963.0, 'mean_token_accuracy': 0.9190744653344154, 'epoch': 2.99}
{'train_runtime': 29637.4415, 'train_samples_per_second': 0.218, 'train_steps_per_second': 0.054, 'train_loss': 0.34627421069174746, 'num_tokens': 6437025.0, 'mean_token_accuracy': 0.9164003014564515, 'epoch': 3.0}
100%|███████████████████████████████████████████████████████████████████| 1614/1614 [8:13:57<00:00, 18.36s/it]
No files have been modified since last commit. Skipping to prevent empty commit.
zsh: killed     python3.11 fine-tune_sfttrainer.py
/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
"""