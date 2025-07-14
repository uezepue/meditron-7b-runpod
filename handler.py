from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ✅ Load once at cold start
tokenizer = None
generator = None

def load_model():
    global tokenizer, generator
    model_name = "epfl-llm/meditron-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

# ✅ Cold start load
load_model()

# ✅ RunPod serverless entrypoint
def handler(event):
    try:
        job_input = event["input"]
        prompt = job_input["prompt"]

        outputs = generator(
            prompt,
            max_length=job_input.get("max_length", 512),
            temperature=job_input.get("temperature", 0.7),
            top_p=job_input.get("top_p", 0.9),
            do_sample=True,
            num_return_sequences=1
        )

        return {"output": outputs[0]["generated_text"]}

    except Exception as e:
        return {"error": str(e)}
