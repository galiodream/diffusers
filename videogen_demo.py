from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video


MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"
PROMPT = "Spiderman is surfing"
OUTPUT_PATH = Path("text_to_video_ms_low_vram_12steps.mp4")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script expects a CUDA GPU.")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Sequential offload saves more VRAM than keeping the whole pipeline on GPU.
    pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")

    generator = torch.Generator(device="cpu").manual_seed(42)

    video_frames = pipe(
        PROMPT,
        num_frames=8,
        num_inference_steps=12,
        height=256,
        width=256,
        guidance_scale=7.5,
        generator=generator,
    ).frames[0]

    video_path = export_to_video(video_frames, output_video_path=str(OUTPUT_PATH), fps=8)
    print(f"saved {video_path}")


if __name__ == "__main__":
    main()
