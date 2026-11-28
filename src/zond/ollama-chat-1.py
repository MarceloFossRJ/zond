from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from dotenv import load_dotenv
import os

load_dotenv()
ollama_model = os.environ['OLLAMA_BASE_URL']

print("Host name: " + ollama_model)

#model = 'qwen3.8:27b-mlx'
#model = 'qwen3.5:9b'
model = 'qwen3-coder:30b'

ollama = OllamaModel(
    model_name=model,
    provider=OllamaProvider(base_url=ollama_model)  # Pointing to Ollama API
)

chat_agent = Agent(ollama)

print(" hey, how can I help you? ")
exit_conditions = (":q", "quit", "exit")

while True:
    query = input("> ")
    if query in exit_conditions:
        break
    else:
        result = chat_agent.run_sync(query)
        print(result.output)

print(" Bye ")