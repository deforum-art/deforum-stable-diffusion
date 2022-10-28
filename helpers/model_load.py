import torch

# Decodes the image without passing through the upscaler. The resulting image will be the same size as the latent
# Thanks to Kevin Turner (https://github.com/keturn) we have a shortcut to look at the decoded image!
def make_linear_decode(model_version, device='cuda:0'):
    v1_4_rgb_latent_factors = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]

    if model_version[:5] == "sd-v1":
        rgb_latent_factors = torch.Tensor(v1_4_rgb_latent_factors).to(device)
    else:
        raise Exception(f"Model name {model_version} not recognized.")

    def linear_decode(latent):
        latent_image = latent.permute(0, 2, 3, 1) @ rgb_latent_factors
        latent_image = latent_image.permute(0, 3, 1, 2)
        return latent_image

    return linear_decode