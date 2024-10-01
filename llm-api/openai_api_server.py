import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

model = "zephyr-7b-beta"
prompt = "Once upon a time"

# create a chat completion
completion = openai.chat.completions.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
# print the completion
print(completion.choices[0].message.content)