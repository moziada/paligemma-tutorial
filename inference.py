import torch
import torch.backends
from PIL import Image
from processing_paligemma import PaliGemmaProcessor
from utils import load_hf_model
from modeling_gemma import PaliGemmaForConditionalGeneration, KVCache

def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def _sample_top_p(probs: torch.Tensor, p: float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - p > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))   # normalize probs back to sum up to 1
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
    
def get_model_inputs(
    processor: PaliGemmaProcessor,
    prompt: str,
    image_file_path: str,
    device: str,
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(prompts, images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(
        model: PaliGemmaForConditionalGeneration,
        processor: PaliGemmaProcessor,
        device: str,
        prompt: str,
        image_path: str,
        max_tokens_to_generate: int,
        temprature: float,
        top_p: float,
        do_sample: bool
):
    model_inputs = get_model_inputs(processor, prompt, image_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs['kv_cache']
        next_token_logits = outputs['logits'][:, -1, :]     # last element in seq_len
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temprature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdims=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # remove batch dim
        generated_tokens.append(next_token)
        if next_token.item() == stop_token:
            break
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1)

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(prompt + decoded)
        
def main(
        model_path: str=None,
        prompt: str=None,
        image_file_path: str=None,
        max_tokens_to_generate: int=100,
        temperature: float=0.8,
        top_p: float=0.9,
        do_sample: bool=False,
        only_cpu: bool=False,
):
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print(f"Device in use: {device}")
    print("Loading Model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running Inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )

if __name__=="__main__":
    MODEL_PATH = "./paligemma-3b-pt-224"
    PROMPT = "this is an image of"
    IMAGE_FILE_PATH = "test_images/34d5b5a3fbaa4b3b5d9487bf924b0145.jpg"
    MAX_TOKENS_TO_GENERATE = 100
    TEMPERATURE = 0.8
    TOP_P = 0.9
    DO_SAMPLE = False
    ONLY_CPU = True

    main(
        model_path=MODEL_PATH,
        prompt=PROMPT,
        image_file_path=IMAGE_FILE_PATH,
        max_tokens_to_generate=MAX_TOKENS_TO_GENERATE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=DO_SAMPLE,
        only_cpu=ONLY_CPU
    )
