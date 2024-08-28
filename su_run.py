from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from lib.predictor import su_inference
from lib.mapper import Mapper

from PIL import Image
import requests
import copy
import torch
import json
import time
import sys
import warnings

# load model
warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

# input image
image = Image.open("su_data/eval_8cls/crowded_1.jpg")

# load query prompt
with open('su_prompts/prompt_8cls.json','r') as f:
    query_dict=json.load(f)
print(f'query_dict:\n{query_dict}\n')

# inference
start_time = time.time()
answer_dict = su_inference(tokenizer, model, image_processor, device, image, query_dict)
end_time = time.time()
execution_time = end_time - start_time
print(f"\nsu_inference with: {execution_time:.2f} sec\n")
print(answer_dict)

# inference with returning a dict of boolean
mapper=Mapper(answer_dict)
bool_dict=mapper.answer2bool()
print(f'bool_dict:\n{bool_dict}\n')