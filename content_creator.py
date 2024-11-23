from transformers import AutoTokenizer,AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

previous_topics = ""

with open('prev_topics.txt','r') as file:
    for row in file:
        previous_topics += row

# Prepare input
prompt = f'Generate a highly detailed and imaginative description of an alternate historical scenario for an image generator. The description should include a specific time period or historical event, an unexpected technological, cultural, or fantastical twist, which should sometimes be futuristic .Specific visual elements like settings, objects, or characters. Make the prompt vivid and specific enough for an AI art generator to create an accurate image. Dont make the prompt longer than 30 tokens, and make it a whole sentence. Avoid repeating ideas already generated. Ideas already generated: {previous_topics}. Only come up with 1 idea and make the structure and the description similar to the previous topics, but with different ideas. Idea or topic in 1 full sentence: $\n'
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Generate text
print("Generating text...")
generate_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=80,  # Use max_new_tokens instead of max_length
    do_sample=True,      # Enable sampling to get varied outputs
    temperature=0.1,     # Control the randomness of predictions
)
output = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

generated_prompt = output.split('$')[1]
generated_sentence = generated_prompt.split('.')[0]

with open('prev_topics.txt','a') as file:
    file.write(generated_sentence.strip() + '\n')

print(generated_sentence)
