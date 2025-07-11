import torch
import os
import json
from tqdm import tqdm
import gc

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import time
import glob as gb


class LLavaAgent:
    def __init__(self, model_path, device='cuda', conv_mode='vicuna_v1', load_8bit=False, load_4bit=True):
        self.device = device
        if torch.device(self.device).index is not None:
            device_map = {'model': torch.device(self.device).index, 'lm_head': torch.device(self.device).index}
        else:
            device_map = 'auto'
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name, device=self.device, device_map=device_map,
            load_8bit=load_8bit, load_4bit=load_4bit)
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.context_len = context_len
        # self.qs = 'Describe this image and its style in a very detailed manner.'
        self.qs = 'Please describe the actual objects in the image in a very detailed manner. Please do not include descriptions related to the focus and bokeh of this image. Please do not include descriptions like the background is blurred.'
        self.conv_mode = conv_mode

        if self.model.config.mm_use_im_start_end:
            self.qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + self.qs
        else:
            self.qs = DEFAULT_IMAGE_TOKEN + '\n' + self.qs

        self.conv = conv_templates[self.conv_mode].copy()
        self.conv.append_message(self.conv.roles[0], self.qs)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(self.device)

    def update_qs(self, qs=None):
        if qs is None:
            qs = self.qs
        else:
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        self.conv = conv_templates[self.conv_mode].copy()
        self.conv.append_message(self.conv.roles[0], qs)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        self.input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(self.device)

    def gen_image_caption(self, imgs, temperature=0.2, top_p=0.7, num_beams=1, qs=None):
        '''
        [PIL.Image, ...]
        '''
        self.update_qs(qs)

        bs = len(imgs)
        input_ids = self.input_ids.repeat(bs, 1)
        img_tensor_list = []
        for image in imgs:
            _image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            img_tensor_list.append(_image_tensor)
        image_tensor = torch.stack(img_tensor_list, dim=0).half().to(self.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=512,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        
        outputs = self.tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)
        # ipdb.set_trace()
        return outputs
        img_captions = []
        for output in outputs:
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip().replace('\n', ' ').replace('\r', ' ')
            img_captions.append(output)
        return outputs

# import ipdb
from transformers import CLIPTextModel, CLIPTokenizer
if __name__ == '__main__':
    llava_agent = LLavaAgent("/data2/cjy/FaithDiff/llava_v1.5-13b/models--liuhaotian--llava-v1.5-13b/snapshots/llava", device='cuda', load_8bit=True, load_4bit=False)
    img = [Image.open('/data2/cjy/RealDeg/real_scene_data/Old_photo_resize/0.jpg'), Image.open('/data2/cjy/RealDeg/real_scene_data/Old_photo_resize/0.jpg')]
    
    caption = llava_agent.gen_image_caption(img, qs='Describe this image and its style in a very detailed manner.')
#     tokenizer = CLIPTokenizer.from_pretrained(
#     '/data/jy/Instruct_Face_restoration/checkpoints/blipdiffusion', subfolder="tokenizer", revision=None
# )
    print(caption)
    # ipdb.set_trace()
    # # ipdb.set_trace()
    # tokenized_prompt = tokenizer('The image features a man with a prominent nose, a thin mustache, and a prominent chin. He has a strong jawline and a prominent forehead. The man is wearing a green shirt and appears to be staring directly into the camera. The close-up shot of his face showcases his distinct facial features, making it a striking and memorable portrait.',
    #                                   padding="max_length",truncation=True,max_length=43,return_tensors="pt",)