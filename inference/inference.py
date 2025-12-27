import torch
from transformers import AutoModelForCausalLM, AutoProcessor

USER_PROMPT = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"
# NOTE: transformers==4.46.3 is recommended for this script
model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"
video_path = "assets/00007869.mp4"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


@torch.inference_mode()
def infer(conversation):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


# Image conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": video_path}},
            {"type": "text", "text": USER_PROMPT},
        ]
    }
]
print("Response:",infer(conversation))


