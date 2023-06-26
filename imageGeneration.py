from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker = False)
pipe = pipe.to("cuda")

prompt = "Astronaut rides horse"
image = pipe(prompt).images[0]
    
image.save("results/astronaut_rides_horse.png")