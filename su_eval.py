import os
import sys
import json
import time
import argparse
import datetime

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from lib.predictor import su_inference
from lib.mapper import Mapper
from lib.eval_utils import *

from PIL import Image
import requests
import copy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score

import sys
import warnings

def generate_default_json_name():
    now = datetime.datetime.now()
    return now.strftime('%m%d_%H%M')

if __name__=='__main__':

    import warnings
    warnings.filterwarnings("ignore")

    # input arg
    parser = argparse.ArgumentParser(description='Run LLaVA model inference.')

    parser.add_argument(
        '--json_file', type=str, 
        required=True, 
        help='Path to the image file'
        )
    parser.add_argument(
        '--prompt_file', type=str, 
        default='./su_prompts/prompt_8cls.json', 
        help='Prompt file path'
        )   
    parser.add_argument(
        '--make_json', type=bool, 
        default=True, 
        help='make an output file'
        )

    input_args = parser.parse_args()

    # load model
    warnings.filterwarnings("ignore")
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()
    
    # load images
    with open(input_args.json_file,'r') as j:
        data_dict=json.load(j)
    
    # load query prompt
    with open(input_args.prompt_file,'r') as f:
        query_dict=json.load(f)
    print("query dict : ")
    for i,j in query_dict.items():
        print(i,'->',j )
    
    recall,precision,accuracy,latency=[],[],[],[]

    result_dict={}

    for sample, gt in tqdm(data_dict.items()):
        # read new image
        image = Image.open(sample)

        # inference with returning a dict of answer text
        start_time = time.time()
        answer_dict = su_inference(tokenizer, model, image_processor, device, image, query_dict)
        end_time = time.time()
        execution_time = end_time - start_time
        latency.append(execution_time)
        # print(f"\n{sample} --> su_inference with: {execution_time:.2f} sec\n")

        # inference with returning a dict of boolean
        mapper=Mapper(answer_dict)
        bool_dict=mapper.answer2bool()

        # merge sub-sce to output dict
        output_dict=mapper.merge_bool()

        # convert bool dict to onehot
        pred=bool2binary(output_dict)

        # record
        result_dict[sample]={
            'answer_dict':answer_dict,
            'bool_dict':bool_dict,
            'pred':pred,
            'ground_truth':gt
        }

    # output a json file
    if input_args.make_json:
        json_path = f'su_data/outputs/llava_ov_8cls_0827.json'
        with open(json_path,'w') as j:
            json.dump(result_dict,j,indent=4)
            print(f'\nrecord results -> {json_path} \n')

    # evaluate
    eval_from_json(json_path)