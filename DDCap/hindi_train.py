import torch 
import re 
import gradio as gr
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 
import PIL

    
device = 'cpu'
encoder_checkpoint = 'google/vit-base-patch16-224'
decoder_checkpoint = 'surajp/gpt2-hindi'
model_checkpoint = 'team-indain-image-caption/hindi-image-captioning'
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)



def predict(image,max_length=64, num_beams=4):
  image= PIL.Image.open(image)
  image = image.convert('RGB')

  image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
  clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
  caption_ids = model.generate(image, max_length = max_length)[0]
  caption_text = clean_text(tokenizer.decode(caption_ids))
  return caption_text 



#input = gr.inputs.Image(label="Image to search", type = 'pil', optional=False)
#output = gr.outputs.Textbox(type="auto",label="Captions")


article = "This HuggingFace Space presents a demo for Image captioning in Hindi built with VIT Encoder and GPT2 Decoder"
title = "Hindi Image Captioning System"
examples = "./example_1.jpg"
print(predict(examples))
#interface = gr.Interface(
#        fn=predict,
#        inputs = input,
#        theme="grass",
#        outputs=output,
#        examples = examples,
#        title=title,
#        description=article,
#    )
#interface.launch(debug=True)