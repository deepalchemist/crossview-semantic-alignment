# -*- coding: utf-8 -*-
# hokkien.ywj@gmail.com @2024-05-05 06:17:42
# Last Change:  2024-05-07 10:47:18

from PIL import Image
import requests

import torch
from transformers import CLIPProcessor, CLIPModel
from mmf.models.clip import build_model

def load_openai_clip():
    # OpenAI github
    ckpt_path = "pretrained/ViT-B-32.pt"
    try:
    # loading JIT archive
        model = torch.jit.load(ckpt_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    clip = build_model(state_dict)
    model = clip.visual
    import pdb; pdb.set_trace()

    # Forward
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print(probs)
    return model

def load_huggingface_clip():
    # Huggingface
    # ['text_model', 'vision_model', 'visual_projection', 'text_projection']
    # hidden_size 768
    model = CLIPModel.from_pretrained("pretrained/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("pretrained/clip-vit-base-patch32")
    config = model.config

    # Forward
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print(probs)
    return model

def init_cross_model(clip):
    from mmf.models.rice.module_crossattn import CrossModel, CrossConfig

    config = CrossConfig()
    config = config.from_dict(
        {
            "use_text": False,
            "use_vision": True
        }
    )
    model = CrossModel(config)
    #model.initialize_from_pretrained(clip)

    # Inputs
    input_ids = None
    q = torch.randn(2, 10, config.hidden_size)
    k = torch.randn(2, 20, config.hidden_size)
    v = k 
    bsz = q.size(0)
    tgt_len = q.size(1) + 1
    src_len = k.size(1)
    self_attention_mask = torch.zeros(
        (bsz, tgt_len),
        dtype=torch.long
    ).bool()
    import pdb; pdb.set_trace()
    cross_attention_mask = torch.ones(
        (bsz, 1, tgt_len, src_len),
        dtype=torch.long
    )

    # Forward
    output = model(
        input_ids,
        q,
        k,
        v,
        self_attention_mask,
        cross_attention_mask,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    import pdb; pdb.set_trace()
    print(output.keys())
    return

if __name__=="__main__":
    clip = load_openai_clip()
    #init_cross_model(clip)
