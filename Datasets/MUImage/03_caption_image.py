from tqdm.auto import tqdm
from PIL import Image as pil
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import json

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

image_files = [str(x) for x in Path("audioset_images").glob("*.jpg")]

max_length = 100
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image_path):
    i_image = pil.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    text = "the image shows"
    inputs = processor(i_image, text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


muimage_captions = {image.split("/")[-1]: predict_step(image) for image in tqdm(image_files)}
json.dump(muimage_captions, open("MUImageImageCaptions.json", "w"))
