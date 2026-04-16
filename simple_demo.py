from pathlib import Path

import torch


if not hasattr(torch.amp, "GradScaler") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
    torch.amp.GradScaler = torch.cuda.amp.GradScaler

from diffusers import DiffusionPipeline


def main():
    prompt = "An image of a squirrel in Picasso style"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"

    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    pipeline = pipeline.to(device)

    image = pipeline(prompt, num_inference_steps=2).images[0]
    output_path = Path("demo_output.png")
    image.save(output_path)
    print(f"saved {output_path.resolve()} with {model_id} on {device}")


if __name__ == "__main__":
    main()
