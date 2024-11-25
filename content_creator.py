from transformers import AutoTokenizer,AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

previous_topics = ""
index = 0

with open('prev_topics.txt','r') as file:
    for row in file:
        previous_topics += '; '
        previous_topics += row
        index += 1
index += 1
# Prepare input
prompt = (
    f"Generate a detailed and imaginative description of a fantastic creature. "
    f"Include its name, physical appearance, unique abilities, habitat, and behavior."
    f" Ensure it’s unlike any known mythical being. Ensure the idea is entirely different from these topics: {previous_topics}. "
    f"Keep it under 200 characters"
    f"Your new idea in maximum of 200 characters: $\n"
)

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Generate text
print("Generating text...")
generate_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,  # Use max_new_tokens instead of max_length
    do_sample=True,      # Enable sampling to get varied outputs
    temperature=0.1,
    no_repeat_ngram_size=1,
)
output = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

generated_text = output.split('$')[1].strip()

with open('prev_topics.txt','a') as file:
    file.write(generated_text.strip().replace('\n',' ') + '\n')

print(generated_text)

from openai import OpenAI
client = OpenAI()

prompt_for_dalle = f'Topic: {generated_text}, instructions: ensure that the animal is its natural habitat. Ensure that there is no text on the generated image'

response = client.images.generate(
  model="dall-e-3",
  prompt=prompt_for_dalle,
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url

import requests

response = requests.get(image_url)

if response.status_code == 200:
    with open("image.png", "wb") as file:
        file.write(response.content)

from PIL import Image

filename = 'image_' + str(index) + '.jpg'

image = Image.open("image.png")
image.save(filename, "JPEG")

from instagrapi import Client

cl = Client()
user = os.getenv("INSTA_USER")
pw = os.getenv("INSTA_PW")
cl.login(user, pw)

# Prepare input
caption_prompt = f"""You are a professional Instagram caption generator. Create a catchy, engaging, and concise Instagram caption that includes:
- A casual tone
- Trending, relevant, popular hashtags at the end of the caption
- Start with the name of the creature
-Ensure you are using a structured format and emojis as well
-Keep it under 300 characters. The topic: {generated_text}
-Your caption: $\n"""
inputs = tokenizer(caption_prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Generate caption
print("Generating caption...")
generate_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,  # Use max_new_tokens instead of max_length
    do_sample=True,      # Enable sampling to get varied outputs
    temperature=0.1,
    no_repeat_ngram_size=1,
)
output = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
caption = output.split('$')[1]
caption = caption + '\n\n This content was generated using AI.'
print(caption)

# Upload photo with AI label
cl.photo_upload(
    path=filename,
    caption=caption,
)