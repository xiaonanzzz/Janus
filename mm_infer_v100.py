import os
import hashlib
import requests
import base64
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

torch.cuda.set_device(1)

class DeepSeekJanusWrapper:

    def __init__(self):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            model_path = "deepseek-ai/Janus-Pro-1B"
            vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
            tokenizer = vl_chat_processor.tokenizer
            # tokenizer = tokenizer.to(dtype=torch.float16)

            vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
            vl_gpt = vl_gpt.to(dtype=torch.float16).cuda().eval()

            self.vl_chat_processor = vl_chat_processor
            self.tokenizer = tokenizer
            self.vl_gpt = vl_gpt

    def url_to_base64(self, url):
        response = requests.get(url)
        response.raise_for_status()
        image_data = response.content
        return base64.b64encode(image_data).decode('utf-8')


    def mm_infer(self, text, images):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            vl_chat_processor = self.vl_chat_processor
            tokenizer = self.tokenizer
            vl_gpt = self.vl_gpt
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{text}",
                    "images": images,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(vl_gpt.device)
            # # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # # run the model to get the response
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            print(f"{prepare_inputs['sft_format'][0]}", answer)
            return answer

image_urls = ['https://blog.boon.so/wp-content/uploads/2024/03/Google-Logo-6-scaled.jpg',]

for i in range(len(image_urls)):
    deepseek_mm = DeepSeekJanusWrapper()
    url = image_urls[i]
    base64_image = deepseek_mm.url_to_base64(url)
    base64_image = f"data:image/jpeg;base64,{base64_image}"
    deepseek_mm.mm_infer("Tell me what is the logo in the image.", [base64_image])
