from vllm import LLM, SamplingParams
from dotenv import load_dotenv

load_dotenv()

prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    "Prove the Lagrange theorem.",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens = 1024)

llm = LLM(model="google/gemma-2b", gpu_memory_utilization = 0.98, max_model_len=8000)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
