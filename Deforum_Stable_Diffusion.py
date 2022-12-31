# %%
# !! {"metadata":{
# !!   "id": "ByGXyiHZWM_q"
# !! }}
"""
# **Deforum Stable Diffusion v0.7**
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer and the [Stability.ai](https://stability.ai/) Team. [K Diffusion](https://github.com/crowsonkb/k-diffusion) by [Katherine Crowson](https://twitter.com/RiversHaveWings). Notebook by [deforum](https://discord.gg/upmXXsrwZc)

[Quick Guide](https://docs.google.com/document/d/1RrQv7FntzOuLg4ohjRZPVL7iptIyBhwwbcEYEW2OfcI/edit?usp=sharing) to Deforum v0.6
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "IJjzzkKlWM_s",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "9461224d-7a56-49f1-fe37-d09bfadc02fb"
# !! }}
#@markdown **NVIDIA GPU**
import subprocess, os, sys
sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(f"{sub_p_res[:-1]}")

# %%
# !! {"metadata":{
# !!   "id": "UA8-efH-WM_t"
# !! }}
"""
# Setup
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "0D2HQO-PWM_t",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 1000,
# !!     "referenced_widgets": [
# !!       "d470cbe9be824963b51f012ca9f4613d",
# !!       "8c602107dd944f0c9f5899002dbf759c",
# !!       "99811478d60f4170a63eeb7dc0753efb",
# !!       "50f554484fd04733a2b38c9ed51f8041",
# !!       "0ee22f1b73304a92b647b857c718a6e8",
# !!       "44f494ba82594a42ba99798a493380c0",
# !!       "0a4ded634ba3459fb43972816b3b6498",
# !!       "003739faad5a4268ae9a60625f5edd3c",
# !!       "0008564262474c75b875ed70adf5f29e",
# !!       "62f3f71650ab4ae68928ccc1e61914f5",
# !!       "d1cbc91985824c1583d76caeda9aa4fc",
# !!       "ba50e03a174e4313a7806360a14a2bf5",
# !!       "9c9487efe1144995bad452a752c1ab85",
# !!       "74805bc3acbe477e9401867419728198",
# !!       "035d6dd789fe4bf5a03c3c1264826bec",
# !!       "bba3735d414d4733a536904e53915da9",
# !!       "7c30fba980994723a692fd5bc1495d4e",
# !!       "a63dc7c4c8cc4a86888f4284948ce0af",
# !!       "9a7a31b5bd7045fea7506bfc17103463",
# !!       "31a3bf2d7e614ce78d529e7964b6b429",
# !!       "ba98eaf102d14fb4b38116b09babe285",
# !!       "70131c31deed45c5a6f082051c77771e",
# !!       "dcef3e167f644e53a2bbb37ca150a993",
# !!       "d6fd5a076f0b48bda2d0b6f9f1fa7a15",
# !!       "aa65c197a9f44281a7185e05a496de00",
# !!       "2d459449fd8c4e8a94da74847f173a0f",
# !!       "e0ea915dae08427ca27ce308bd977d13",
# !!       "5bb163fcbe994919b2b6ab195f0b7b4f",
# !!       "32229b2d58e64a66b225c60d04b1d381",
# !!       "8e457f0052a94bb8b07fe550964c1915",
# !!       "3c6996a280084f5989efa8854ea8cdd9",
# !!       "3b7dddb4119e4a2d8bff43ad0018fb61",
# !!       "ae8f37175f024621ad4c8f6e4c148f57",
# !!       "10ca0c323d37456bbd66e8b37f9c5a64",
# !!       "160a8b107d59408289ccb3865bb2043e",
# !!       "cfd1e8459ca2480186acbb28fb92d1ae",
# !!       "29ea3d0ed04b414fbb2c4508867fff78",
# !!       "f414a915b3fe40ef96bda917582b89cb",
# !!       "a46e2526eff64c2a8cb165bb6c228e99",
# !!       "6caa52cca8a445d08d10b68d37a8e04c",
# !!       "d13f331ace0a43c4ad133428f92eab2e",
# !!       "77f55ca0f83643bcb0615063958a23ab",
# !!       "5572821b3ee241098890ce482fa8eef4",
# !!       "fff301b17eeb409fb5224ca50cc49c0c",
# !!       "3d9aa103cc454765b2a683daf9e6d00d",
# !!       "425bb4cd6fbe4c7e96dc9d092cc835c7",
# !!       "f846c5e698d04b27929bbe14fcf2e46f",
# !!       "17581be4595d4f489ac70e566d09eab8",
# !!       "2e6bf76dd5094af3ba7d5add5b400f81",
# !!       "cc501b19bb954c5f8e11a8ad697b2e8c",
# !!       "27a4dbc7c0ee4ec99cf2ff255f531f8b",
# !!       "cdbe671e071a434c84dde408b8ac9d1c",
# !!       "4d8a05b94cfd45f29aef0c8df09d26de",
# !!       "8334c21ae5224b85a34b9275fdda1071",
# !!       "84624c26638c4449a9a6fc206bd4c5f4",
# !!       "abdcbdf1b57b4bee8e5fcd06e6c7c7da",
# !!       "c0a88d752aa041aabff7c8f25aa3671b",
# !!       "d5676003055046d8a1c2c91567c68233",
# !!       "19cc80a7cba24d3bbe0179fca7481f5c",
# !!       "8a9133112bf84e2293154168d89833db",
# !!       "cc1a910b97a84cc19ae7a8fba3a7bb8c",
# !!       "f11ef36637484e45ab6adcb7e9dc2fec",
# !!       "d9fe04ff764c422185e8bab3eeb98148",
# !!       "3b3b457e9f7b4dc0b8d1a6fd13030518",
# !!       "1a51a3e5d05d482e935dfb9c5efef470",
# !!       "cc9694cb995e42c68392b8f2c8130253"
# !!     ]
# !!   },
# !!   "outputId": "086f5210-5eb6-4c03-d3ad-55834a15fa36"
# !! }}

import subprocess, time, gc, os, sys

def setup_environment():
    start_time = time.time()
    print_subprocess = False
    use_xformers_for_colab = True
    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'
    if 'google.colab' in str(ipy):
        print("..setting up environment")
        
        all_process = [
            ['pip', 'install', 'torch==1.12.1+cu113', 'torchvision==0.13.1+cu113', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'],
            ['pip', 'install', 'omegaconf==2.2.3', 'einops==0.4.1', 'pytorch-lightning==1.7.4', 'torchmetrics==0.9.3', 'torchtext==0.13.1', 'transformers==4.21.2', 'safetensors', 'kornia==0.6.7'],
            ['git', 'clone', 'https://github.com/deforum-art/deforum-stable-diffusion'],
            ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq','scikit-learn','torchsde','open-clip-torch', 'numpngw'],
        ]
        for process in all_process:
            running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
            if print_subprocess:
                print(running)
        with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
            f.write('')
        sys.path.extend([
            'deforum-stable-diffusion/',
            'deforum-stable-diffusion/src',
        ])
        if use_xformers_for_colab:

            print("..installing xformers")

            all_process = [['pip', 'install', 'triton==2.0.0.dev20220701']]
            for process in all_process:
                running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                if print_subprocess:
                    print(running)

            v_card_name = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
            if 't4' in v_card_name.lower():
                name_to_download = 'T4'
            elif 'v100' in v_card_name.lower():
                name_to_download = 'V100'
            elif 'a100' in v_card_name.lower():
                name_to_download = 'A100'
            elif 'p100' in v_card_name.lower():
                name_to_download = 'P100'
            elif 'a4000' in v_card_name.lower():
                name_to_download = 'Non-Colab/Paperspace/A4000'
            elif 'p5000' in v_card_name.lower():
                name_to_download = 'Non-Colab/Paperspace/P5000'
            elif 'quadro m4000' in v_card_name.lower():
                name_to_download = 'Non-Colab/Paperspace/Quadro M4000'
            elif 'rtx 4000' in v_card_name.lower():
                name_to_download = 'Non-Colab/Paperspace/RTX 4000'
            elif 'rtx 5000' in v_card_name.lower():
                name_to_download = 'Non-Colab/Paperspace/RTX 5000'
            else:
                print(v_card_name + ' is currently not supported with xformers flash attention in deforum!')

            if 'Non-Colab' in name_to_download:
                x_ver = 'xformers-0.0.14.dev0-cp39-cp39-linux_x86_64.whl'
            else:
                x_ver = 'xformers-0.0.13.dev0-py3-none-any.whl'

            x_link = 'https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/' + name_to_download + '/' + x_ver

            all_process = [
                ['wget', '--no-verbose', '--no-clobber', x_link],
                ['pip', 'install', x_ver],
            ]

            for process in all_process:
                running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                if print_subprocess:
                    print(running)
    else:
        sys.path.extend([
            'src'
        ])
    end_time = time.time()
    print(f"..environment set up in {end_time-start_time:.0f} seconds")
    return

setup_environment()

import torch
import random
import clip
from IPython import display
from types import SimpleNamespace
from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import make_linear_decode, load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model

#@markdown **Path Setup**

def Root():
    models_path = "models" #@param {type:"string"}
    configs_path = "configs" #@param {type:"string"}
    output_path = "output" #@param {type:"string"}
    mount_google_drive = True #@param {type:"boolean"}
    models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
    output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}

    #@markdown **Model Setup**
    model_config = "v1-inference.yaml" #@param ["custom","v2-inference.yaml","v1-inference.yaml"]
    model_checkpoint =  "v1-5-pruned-emaonly.ckpt" #@param ["custom","512-base-ema.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
    custom_config_path = "" #@param {type:"string"}
    custom_checkpoint_path = "" #@param {type:"string"}
    half_precision = True
    return locals()

root = Root()
root = SimpleNamespace(**root)

root.models_path, root.output_path = get_model_output_paths(root)
root.model, root.device = load_model(root, 
                                    load_on_run_all=True
                                    , 
                                    check_sha256=True
                                    )

# %%
# !! {"metadata":{
# !!   "id": "6JxwhBwtWM_t"
# !! }}
"""
# Settings
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "E0tJVYA4WM_u"
# !! }}
def DeforumAnimArgs():

    #@markdown ####**Animation:**
    animation_mode = 'None' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 1000 #@param {type:"number"}
    border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}

    #@markdown ####**Motion Parameters:**
    angle = "0:(0)"#@param {type:"string"}
    zoom = "0:(1.04)"#@param {type:"string"}
    translation_x = "0:(10*sin(2*3.14*t/10))"#@param {type:"string"}
    translation_y = "0:(0)"#@param {type:"string"}
    translation_z = "0:(10)"#@param {type:"string"}
    rotation_3d_x = "0:(0)"#@param {type:"string"}
    rotation_3d_y = "0:(0)"#@param {type:"string"}
    rotation_3d_z = "0:(0)"#@param {type:"string"}
    flip_2d_perspective = False #@param {type:"boolean"}
    perspective_flip_theta = "0:(0)"#@param {type:"string"}
    perspective_flip_phi = "0:(t%15)"#@param {type:"string"}
    perspective_flip_gamma = "0:(0)"#@param {type:"string"}
    perspective_flip_fv = "0:(53)"#@param {type:"string"}
    noise_schedule = "0: (0.02)"#@param {type:"string"}
    strength_schedule = "0: (0.65)"#@param {type:"string"}
    contrast_schedule = "0: (1.0)"#@param {type:"string"}

    #@markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3#@param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40#@param {type:"number"}
    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    video_init_path ='/content/video_in.mp4'#@param {type:"string"}
    extract_nth_frame = 1#@param {type:"number"}
    overwrite_extracted_frames = True #@param {type:"boolean"}
    use_mask_video = False #@param {type:"boolean"}
    video_mask_path ='/content/video_in.mp4'#@param {type:"string"}

    #@markdown ####**Interpolation:**
    interpolate_key_frames = False #@param {type:"boolean"}
    interpolate_x_frames = 4 #@param {type:"number"}
    
    #@markdown ####**Resume Animation:**
    resume_from_timestring = False #@param {type:"boolean"}
    resume_timestring = "20220829210106" #@param {type:"string"}

    return locals()

# %%
# !! {"metadata":{
# !!   "id": "i9fly1RIWM_u"
# !! }}
prompts = [
    "a beautiful lake by Asher Brown Durand, trending on Artstation", # the first prompt I want
    "a beautiful portrait of a woman by Artgerm, trending on Artstation", # the second prompt I want
    #"this prompt I don't want it I commented it out",
    #"a nousr robot, trending on Artstation", # use "nousr robot" with the robot diffusion model (see model_checkpoint setting)
    #"touhou 1girl komeiji_koishi portrait, green hair", # waifu diffusion prompts can use danbooru tag groups (see model_checkpoint)
    #"this prompt has weights if prompt weighting enabled:2 can also do negative:-2", # (see prompt_weighting)
]

animation_prompts = {
    0: "a beautiful apple, trending on Artstation",
    20: "a beautiful banana, trending on Artstation",
    30: "a beautiful coconut, trending on Artstation",
    40: "a beautiful durian, trending on Artstation",
}

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "XVzhbmizWM_u",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 1000,
# !!     "referenced_widgets": [
# !!       "7d9f0925c53a4f85b52f529fe3252dd6",
# !!       "a3be7e7b16cb468e80df3fb6bdbf28f4",
# !!       "585f407717ce4bc5863b36ea711665f0",
# !!       "41008ab04bf5458095e981f459fdc39f",
# !!       "bb4e64d0d5624fafaafa1e8248c40a0e",
# !!       "e36c3eeb5a2e41edab64ee6fd36e5a05",
# !!       "e168704fa5db46cbad1ff9ddac9b000d",
# !!       "37d78a2d84dd48f4b6c1594719bf9eba",
# !!       "fd9670ff3d4c46aeab0cee66de147c16",
# !!       "5873c4d2d9cd4a5d8037d67ac8bf6641",
# !!       "e3e7da82b11945038a6d57ac9d182b7a",
# !!       "7a80a7c5f9a64d158ad805acde075d7e",
# !!       "984317e9f55f4deeb0c00ca64a559480",
# !!       "9e6d2b01e9d74051bf04e31f12c53585",
# !!       "f3147510bd544a2c807ed01d9a48b7b6",
# !!       "13052fc82baf4995b49ef3f5c40027da",
# !!       "fe888d0adae349b9a2a36bd252c20e69",
# !!       "c4e2375486c44f6a91cc0fde39d47080",
# !!       "4dfec0023a6941d281f738fdc7233c87",
# !!       "19e1665e53814e1bbf9205c298f6c53b",
# !!       "033adb5cd71043008214dde61510261e",
# !!       "62a18332f6784d05aa2ebb6376563b9b"
# !!     ]
# !!   },
# !!   "outputId": "ecab9ebb-e40d-4bd1-ce13-87e20bff2575"
# !! }}
#@markdown **Load Settings**
override_settings_with_file = False #@param {type:"boolean"}
settings_file = "custom" #@param ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
custom_settings_file = "/content/drive/MyDrive/Settings.txt"#@param {type:"string"}

def DeforumArgs():
    #@markdown **Image Settings**
    W = 512 #@param
    H = 512 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    #@markdown **Sampling Settings**
    seed = -1 #@param
    sampler = 'euler_ancestral' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
    steps = 80 #@param
    scale = 7 #@param
    ddim_eta = 0.0 #@param
    dynamic_threshold = None
    static_threshold = None   

    #@markdown **Save & Display Settings**
    save_samples = True #@param {type:"boolean"}
    save_settings = True #@param {type:"boolean"}
    display_samples = True #@param {type:"boolean"}
    save_sample_per_step = False #@param {type:"boolean"}
    show_sample_per_step = False #@param {type:"boolean"}

    #@markdown **Prompt Settings**
    prompt_weighting = True #@param {type:"boolean"}
    normalize_prompt_weights = True #@param {type:"boolean"}
    log_weighted_subprompts = False #@param {type:"boolean"}

    #@markdown **Batch Settings**
    n_batch = 1 #@param
    batch_name = "StableFun" #@param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png" #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter" #@param ["iter","fixed","random"]
    make_grid = False #@param {type:"boolean"}
    grid_rows = 2 #@param 
    outdir = get_output_folder(root.output_path, batch_name)

    #@markdown **Init Settings**
    use_init = False #@param {type:"boolean"}
    strength = 0.0 #@param {type:"number"}
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False #@param {type:"boolean"}
    use_alpha_as_mask = False # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" #@param {type:"string"}
    invert_mask = False #@param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  #@param {type:"number"}
    mask_contrast_adjust = 1.0  #@param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5 # {type:"number"}

    #@markdown **Exposure/Contrast Conditional Settings**
    mean_scale = 0 #@param {type:"number"}
    var_scale = 0 #@param {type:"number"}
    exposure_scale = 0 #@param {type:"number"}
    exposure_target = 0.5 #@param {type:"number"}

    #@markdown **Color Match Conditional Settings**
    colormatch_scale = 0 #@param {type:"number"}
    colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png" #@param {type:"string"}
    colormatch_n_colors = 4 #@param {type:"number"}
    ignore_sat_weight = 0 #@param {type:"number"}

    #@markdown **CLIP\Aesthetics Conditional Settings**
    clip_name = 'ViT-L/14' #@param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
    clip_scale = 0 #@param {type:"number"}
    aesthetics_scale = 0 #@param {type:"number"}
    cutn = 1 #@param {type:"number"}
    cut_pow = 0.0001 #@param {type:"number"}

    #@markdown **Other Conditional Settings**
    init_mse_scale = 0 #@param {type:"number"}
    init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}

    blue_scale = 0 #@param {type:"number"}
    
    #@markdown **Conditional Gradient Settings**
    gradient_wrt = 'x0_pred' #@param ["x", "x0_pred"]
    gradient_add_to = 'both' #@param ["cond", "uncond", "both"]
    decode_method = 'linear' #@param ["autoencoder","linear"]
    grad_threshold_type = 'dynamic' #@param ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = 0.2 #@param {type:"number"}
    clamp_start = 0.2 #@param
    clamp_stop = 0.01 #@param
    grad_inject_timing = list(range(1,10)) #@param

    #@markdown **Speed vs VRAM Settings**
    cond_uncond_sync = True #@param {type:"boolean"}

    n_samples = 1 # doesnt do anything
    precision = 'autocast' 
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None

    return locals()

args_dict = DeforumArgs()
anim_args_dict = DeforumAnimArgs()

if override_settings_with_file:
    load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=False)

args = SimpleNamespace(**args_dict)
anim_args = SimpleNamespace(**anim_args_dict)

args.timestring = time.strftime('%Y%m%d%H%M%S')
args.strength = max(0.0, min(1.0, args.strength))

# Load clip model if using clip guidance
if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
    root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
    if (args.aesthetics_scale > 0):
        root.aesthetics_model = load_aesthetics_model(args, root)

if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)
if not args.use_init:
    args.init_image = None
if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
    print(f"Init images aren't supported with PLMS yet, switching to KLMS")
    args.sampler = 'klms'
if args.sampler != 'ddim':
    args.ddim_eta = 0

if anim_args.animation_mode == 'None':
    anim_args.max_frames = 1
elif anim_args.animation_mode == 'Video Input':
    args.use_init = True

# clean up unused memory
gc.collect()
torch.cuda.empty_cache()

# dispatch to appropriate renderer
if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
    render_animation(args, anim_args, animation_prompts, root)
elif anim_args.animation_mode == 'Video Input':
    render_input_video(args, anim_args, animation_prompts, root)
elif anim_args.animation_mode == 'Interpolation':
    render_interpolation(args, anim_args, animation_prompts, root)
else:
    render_image_batch(args, prompts, root)

# %%
# !! {"metadata":{
# !!   "id": "gJ88kZ2-WM_v"
# !! }}
"""
# Create Video From Frames
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "XQGeqaGAWM_v"
# !! }}
skip_video_for_run_all = True #@param {type: 'boolean'}
fps = 12 #@param {type:"number"}
#@markdown **Manual Settings**
use_manual_settings = False #@param {type:"boolean"}
image_path = "/content/drive/MyDrive/AI/StableDiffusion/2022-09/20220903000939_%05d.png" #@param {type:"string"}
mp4_path = "/content/drive/MyDrive/AI/StableDiffusion/2022-09/20220903000939.mp4" #@param {type:"string"}
render_steps = False  #@param {type: 'boolean'}
path_name_modifier = "x0_pred" #@param ["x0_pred","x"]
make_gif = False

if skip_video_for_run_all == True:
    print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
else:
    import os
    import subprocess
    from base64 import b64encode

    print(f"{image_path} -> {mp4_path}")

    if use_manual_settings:
        max_frames = "200" #@param {type:"string"}
    else:
        if render_steps: # render steps from a single image
            fname = f"{path_name_modifier}_%05d.png"
            all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir,d))]
            newest_dir = max(all_step_dirs, key=os.path.getmtime)
            image_path = os.path.join(newest_dir, fname)
            print(f"Reading images from {image_path}")
            mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
            max_frames = str(args.steps)
        else: # render images for a video
            image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
            mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
            max_frames = str(anim_args.max_frames)

    # make video
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', image_path,
        '-frames:v', max_frames,
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        '-pattern_type', 'sequence',
        mp4_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)

    mp4 = open(mp4_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display.display(display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )
    
    if make_gif:
         gif_path = os.path.splitext(mp4_path)[0]+'.gif'
         cmd_gif = [
             'ffmpeg',
             '-y',
             '-i', mp4_path,
             '-r', str(fps),
             gif_path
         ]
         process_gif = subprocess.Popen(cmd_gif, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "MMpAcyrYWM_v"
# !! }}
skip_disconnect_for_run_all = True #@param {type: 'boolean'}

if skip_disconnect_for_run_all == True:
    print('Skipping disconnect, uncheck skip_disconnect_for_run_all if you want to run it')
else:
    from google.colab import runtime
    runtime.unassign()

# %%
# !! {"main_metadata":{
# !!   "kernelspec": {
# !!     "display_name": "Python 3.10.6 ('dsd')",
# !!     "language": "python",
# !!     "name": "python3"
# !!   },
# !!   "language_info": {
# !!     "codemirror_mode": {
# !!       "name": "ipython",
# !!       "version": 3
# !!     },
# !!     "file_extension": ".py",
# !!     "mimetype": "text/x-python",
# !!     "name": "python",
# !!     "nbconvert_exporter": "python",
# !!     "pygments_lexer": "ipython3",
# !!     "version": "3.10.6"
# !!   },
# !!   "orig_nbformat": 4,
# !!   "vscode": {
# !!     "interpreter": {
# !!       "hash": "b7e04c8a9537645cbc77fa0cbde8069bc94e341b0d5ced104651213865b24e58"
# !!     }
# !!   },
# !!   "colab": {
# !!     "provenance": [],
# !!     "machine_shape": "hm"
# !!   },
# !!   "accelerator": "GPU",
# !!   "gpuClass": "standard",
# !!   "widgets": {
# !!     "application/vnd.jupyter.widget-state+json": {
# !!       "d470cbe9be824963b51f012ca9f4613d": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_8c602107dd944f0c9f5899002dbf759c",
# !!             "IPY_MODEL_99811478d60f4170a63eeb7dc0753efb",
# !!             "IPY_MODEL_50f554484fd04733a2b38c9ed51f8041"
# !!           ],
# !!           "layout": "IPY_MODEL_0ee22f1b73304a92b647b857c718a6e8"
# !!         }
# !!       },
# !!       "8c602107dd944f0c9f5899002dbf759c": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_44f494ba82594a42ba99798a493380c0",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_0a4ded634ba3459fb43972816b3b6498",
# !!           "value": "Downloading vocab.json: 100%"
# !!         }
# !!       },
# !!       "99811478d60f4170a63eeb7dc0753efb": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_003739faad5a4268ae9a60625f5edd3c",
# !!           "max": 961143,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_0008564262474c75b875ed70adf5f29e",
# !!           "value": 961143
# !!         }
# !!       },
# !!       "50f554484fd04733a2b38c9ed51f8041": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_62f3f71650ab4ae68928ccc1e61914f5",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_d1cbc91985824c1583d76caeda9aa4fc",
# !!           "value": " 939k/939k [00:01&lt;00:00, 886kB/s]"
# !!         }
# !!       },
# !!       "0ee22f1b73304a92b647b857c718a6e8": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "44f494ba82594a42ba99798a493380c0": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "0a4ded634ba3459fb43972816b3b6498": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "003739faad5a4268ae9a60625f5edd3c": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "0008564262474c75b875ed70adf5f29e": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "62f3f71650ab4ae68928ccc1e61914f5": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "d1cbc91985824c1583d76caeda9aa4fc": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "ba50e03a174e4313a7806360a14a2bf5": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_9c9487efe1144995bad452a752c1ab85",
# !!             "IPY_MODEL_74805bc3acbe477e9401867419728198",
# !!             "IPY_MODEL_035d6dd789fe4bf5a03c3c1264826bec"
# !!           ],
# !!           "layout": "IPY_MODEL_bba3735d414d4733a536904e53915da9"
# !!         }
# !!       },
# !!       "9c9487efe1144995bad452a752c1ab85": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_7c30fba980994723a692fd5bc1495d4e",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_a63dc7c4c8cc4a86888f4284948ce0af",
# !!           "value": "Downloading merges.txt: 100%"
# !!         }
# !!       },
# !!       "74805bc3acbe477e9401867419728198": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_9a7a31b5bd7045fea7506bfc17103463",
# !!           "max": 524619,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_31a3bf2d7e614ce78d529e7964b6b429",
# !!           "value": 524619
# !!         }
# !!       },
# !!       "035d6dd789fe4bf5a03c3c1264826bec": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_ba98eaf102d14fb4b38116b09babe285",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_70131c31deed45c5a6f082051c77771e",
# !!           "value": " 512k/512k [00:01&lt;00:00, 529kB/s]"
# !!         }
# !!       },
# !!       "bba3735d414d4733a536904e53915da9": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "7c30fba980994723a692fd5bc1495d4e": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "a63dc7c4c8cc4a86888f4284948ce0af": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "9a7a31b5bd7045fea7506bfc17103463": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "31a3bf2d7e614ce78d529e7964b6b429": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "ba98eaf102d14fb4b38116b09babe285": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "70131c31deed45c5a6f082051c77771e": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "dcef3e167f644e53a2bbb37ca150a993": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_d6fd5a076f0b48bda2d0b6f9f1fa7a15",
# !!             "IPY_MODEL_aa65c197a9f44281a7185e05a496de00",
# !!             "IPY_MODEL_2d459449fd8c4e8a94da74847f173a0f"
# !!           ],
# !!           "layout": "IPY_MODEL_e0ea915dae08427ca27ce308bd977d13"
# !!         }
# !!       },
# !!       "d6fd5a076f0b48bda2d0b6f9f1fa7a15": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_5bb163fcbe994919b2b6ab195f0b7b4f",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_32229b2d58e64a66b225c60d04b1d381",
# !!           "value": "Downloading special_tokens_map.json: 100%"
# !!         }
# !!       },
# !!       "aa65c197a9f44281a7185e05a496de00": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_8e457f0052a94bb8b07fe550964c1915",
# !!           "max": 389,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_3c6996a280084f5989efa8854ea8cdd9",
# !!           "value": 389
# !!         }
# !!       },
# !!       "2d459449fd8c4e8a94da74847f173a0f": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_3b7dddb4119e4a2d8bff43ad0018fb61",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_ae8f37175f024621ad4c8f6e4c148f57",
# !!           "value": " 389/389 [00:00&lt;00:00, 24.6kB/s]"
# !!         }
# !!       },
# !!       "e0ea915dae08427ca27ce308bd977d13": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "5bb163fcbe994919b2b6ab195f0b7b4f": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "32229b2d58e64a66b225c60d04b1d381": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "8e457f0052a94bb8b07fe550964c1915": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "3c6996a280084f5989efa8854ea8cdd9": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "3b7dddb4119e4a2d8bff43ad0018fb61": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "ae8f37175f024621ad4c8f6e4c148f57": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "10ca0c323d37456bbd66e8b37f9c5a64": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_160a8b107d59408289ccb3865bb2043e",
# !!             "IPY_MODEL_cfd1e8459ca2480186acbb28fb92d1ae",
# !!             "IPY_MODEL_29ea3d0ed04b414fbb2c4508867fff78"
# !!           ],
# !!           "layout": "IPY_MODEL_f414a915b3fe40ef96bda917582b89cb"
# !!         }
# !!       },
# !!       "160a8b107d59408289ccb3865bb2043e": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_a46e2526eff64c2a8cb165bb6c228e99",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_6caa52cca8a445d08d10b68d37a8e04c",
# !!           "value": "Downloading tokenizer_config.json: 100%"
# !!         }
# !!       },
# !!       "cfd1e8459ca2480186acbb28fb92d1ae": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_d13f331ace0a43c4ad133428f92eab2e",
# !!           "max": 905,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_77f55ca0f83643bcb0615063958a23ab",
# !!           "value": 905
# !!         }
# !!       },
# !!       "29ea3d0ed04b414fbb2c4508867fff78": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_5572821b3ee241098890ce482fa8eef4",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_fff301b17eeb409fb5224ca50cc49c0c",
# !!           "value": " 905/905 [00:00&lt;00:00, 38.6kB/s]"
# !!         }
# !!       },
# !!       "f414a915b3fe40ef96bda917582b89cb": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "a46e2526eff64c2a8cb165bb6c228e99": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "6caa52cca8a445d08d10b68d37a8e04c": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "d13f331ace0a43c4ad133428f92eab2e": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "77f55ca0f83643bcb0615063958a23ab": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "5572821b3ee241098890ce482fa8eef4": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "fff301b17eeb409fb5224ca50cc49c0c": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "3d9aa103cc454765b2a683daf9e6d00d": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_425bb4cd6fbe4c7e96dc9d092cc835c7",
# !!             "IPY_MODEL_f846c5e698d04b27929bbe14fcf2e46f",
# !!             "IPY_MODEL_17581be4595d4f489ac70e566d09eab8"
# !!           ],
# !!           "layout": "IPY_MODEL_2e6bf76dd5094af3ba7d5add5b400f81"
# !!         }
# !!       },
# !!       "425bb4cd6fbe4c7e96dc9d092cc835c7": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_cc501b19bb954c5f8e11a8ad697b2e8c",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_27a4dbc7c0ee4ec99cf2ff255f531f8b",
# !!           "value": "Downloading config.json: 100%"
# !!         }
# !!       },
# !!       "f846c5e698d04b27929bbe14fcf2e46f": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_cdbe671e071a434c84dde408b8ac9d1c",
# !!           "max": 4519,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_4d8a05b94cfd45f29aef0c8df09d26de",
# !!           "value": 4519
# !!         }
# !!       },
# !!       "17581be4595d4f489ac70e566d09eab8": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_8334c21ae5224b85a34b9275fdda1071",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_84624c26638c4449a9a6fc206bd4c5f4",
# !!           "value": " 4.41k/4.41k [00:00&lt;00:00, 254kB/s]"
# !!         }
# !!       },
# !!       "2e6bf76dd5094af3ba7d5add5b400f81": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "cc501b19bb954c5f8e11a8ad697b2e8c": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "27a4dbc7c0ee4ec99cf2ff255f531f8b": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "cdbe671e071a434c84dde408b8ac9d1c": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "4d8a05b94cfd45f29aef0c8df09d26de": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "8334c21ae5224b85a34b9275fdda1071": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "84624c26638c4449a9a6fc206bd4c5f4": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "abdcbdf1b57b4bee8e5fcd06e6c7c7da": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_c0a88d752aa041aabff7c8f25aa3671b",
# !!             "IPY_MODEL_d5676003055046d8a1c2c91567c68233",
# !!             "IPY_MODEL_19cc80a7cba24d3bbe0179fca7481f5c"
# !!           ],
# !!           "layout": "IPY_MODEL_8a9133112bf84e2293154168d89833db"
# !!         }
# !!       },
# !!       "c0a88d752aa041aabff7c8f25aa3671b": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_cc1a910b97a84cc19ae7a8fba3a7bb8c",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_f11ef36637484e45ab6adcb7e9dc2fec",
# !!           "value": "Downloading pytorch_model.bin: 100%"
# !!         }
# !!       },
# !!       "d5676003055046d8a1c2c91567c68233": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_d9fe04ff764c422185e8bab3eeb98148",
# !!           "max": 1710671599,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_3b3b457e9f7b4dc0b8d1a6fd13030518",
# !!           "value": 1710671599
# !!         }
# !!       },
# !!       "19cc80a7cba24d3bbe0179fca7481f5c": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_1a51a3e5d05d482e935dfb9c5efef470",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_cc9694cb995e42c68392b8f2c8130253",
# !!           "value": " 1.59G/1.59G [00:23&lt;00:00, 74.6MB/s]"
# !!         }
# !!       },
# !!       "8a9133112bf84e2293154168d89833db": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "cc1a910b97a84cc19ae7a8fba3a7bb8c": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "f11ef36637484e45ab6adcb7e9dc2fec": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "d9fe04ff764c422185e8bab3eeb98148": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "3b3b457e9f7b4dc0b8d1a6fd13030518": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "1a51a3e5d05d482e935dfb9c5efef470": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "cc9694cb995e42c68392b8f2c8130253": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "7d9f0925c53a4f85b52f529fe3252dd6": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_a3be7e7b16cb468e80df3fb6bdbf28f4",
# !!             "IPY_MODEL_585f407717ce4bc5863b36ea711665f0",
# !!             "IPY_MODEL_41008ab04bf5458095e981f459fdc39f"
# !!           ],
# !!           "layout": "IPY_MODEL_bb4e64d0d5624fafaafa1e8248c40a0e"
# !!         }
# !!       },
# !!       "a3be7e7b16cb468e80df3fb6bdbf28f4": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_e36c3eeb5a2e41edab64ee6fd36e5a05",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_e168704fa5db46cbad1ff9ddac9b000d",
# !!           "value": "100%"
# !!         }
# !!       },
# !!       "585f407717ce4bc5863b36ea711665f0": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_37d78a2d84dd48f4b6c1594719bf9eba",
# !!           "max": 80,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_fd9670ff3d4c46aeab0cee66de147c16",
# !!           "value": 80
# !!         }
# !!       },
# !!       "41008ab04bf5458095e981f459fdc39f": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_5873c4d2d9cd4a5d8037d67ac8bf6641",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_e3e7da82b11945038a6d57ac9d182b7a",
# !!           "value": " 80/80 [00:16&lt;00:00,  6.14it/s]"
# !!         }
# !!       },
# !!       "bb4e64d0d5624fafaafa1e8248c40a0e": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "e36c3eeb5a2e41edab64ee6fd36e5a05": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "e168704fa5db46cbad1ff9ddac9b000d": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "37d78a2d84dd48f4b6c1594719bf9eba": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "fd9670ff3d4c46aeab0cee66de147c16": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "5873c4d2d9cd4a5d8037d67ac8bf6641": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "e3e7da82b11945038a6d57ac9d182b7a": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "7a80a7c5f9a64d158ad805acde075d7e": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_984317e9f55f4deeb0c00ca64a559480",
# !!             "IPY_MODEL_9e6d2b01e9d74051bf04e31f12c53585",
# !!             "IPY_MODEL_f3147510bd544a2c807ed01d9a48b7b6"
# !!           ],
# !!           "layout": "IPY_MODEL_13052fc82baf4995b49ef3f5c40027da"
# !!         }
# !!       },
# !!       "984317e9f55f4deeb0c00ca64a559480": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_fe888d0adae349b9a2a36bd252c20e69",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_c4e2375486c44f6a91cc0fde39d47080",
# !!           "value": "100%"
# !!         }
# !!       },
# !!       "9e6d2b01e9d74051bf04e31f12c53585": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_4dfec0023a6941d281f738fdc7233c87",
# !!           "max": 80,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_19e1665e53814e1bbf9205c298f6c53b",
# !!           "value": 80
# !!         }
# !!       },
# !!       "f3147510bd544a2c807ed01d9a48b7b6": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_033adb5cd71043008214dde61510261e",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_62a18332f6784d05aa2ebb6376563b9b",
# !!           "value": " 80/80 [00:13&lt;00:00,  6.14it/s]"
# !!         }
# !!       },
# !!       "13052fc82baf4995b49ef3f5c40027da": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "fe888d0adae349b9a2a36bd252c20e69": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "c4e2375486c44f6a91cc0fde39d47080": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "4dfec0023a6941d281f738fdc7233c87": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "19e1665e53814e1bbf9205c298f6c53b": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "033adb5cd71043008214dde61510261e": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "62a18332f6784d05aa2ebb6376563b9b": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       }
# !!     }
# !!   }
# !! }}
