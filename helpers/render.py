import os
import json
from IPython import display
import random
from torchvision.utils import make_grid
from einops import rearrange
import pandas as pd
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from PIL import Image, ImageOps
import pathlib
import torchvision.transforms as T

from .generate import generate, add_noise
from .prompt import sanitize
from .animation import DeformAnimKeys, sample_from_cv2, sample_to_cv2, anim_frame_warp, vid2frames
from .depth import DepthModel
from .colors import maintain_colors
from .load_images import prepare_overlay_mask
from .hybrid_video import hybrid_generation, hybrid_composite
from .hybrid_video import get_matrix_for_hybrid_motion, get_matrix_for_hybrid_motion_prev, get_flow_for_hybrid_motion, get_flow_for_hybrid_motion_prev, image_transform_ransac, image_transform_optical_flow

try:
    from numpngw import write_png
except ModuleNotFoundError:
    print(ModuleNotFoundError)
    import subprocess
    running = subprocess.run(['pip', 'install', 'numpngw'],stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(running)
    from numpngw import write_png

#import tifffile # Un-comment to save 32bpc TIFF images too. Also un-comment line within 'def save_8_16_or_32bpc_image()'

# This function converts the image to 8bpc (if it isn't already) to display it on browser.
def convert_image_to_8bpc(image, bit_depth_output): 
    if bit_depth_output == 16:
        image = image / 256
        image = Image.fromarray(image.astype('uint8'))
    elif bit_depth_output == 32:
        image = np.clip(image * 256, 0, 255) # Clip values below 0 and above 255 (but those values ARE PRESENT in the EXRs)
        image = Image.fromarray(image.astype('uint8'))
    return image

# This function saves the image to file, depending on bitrate. At 8bpc PIL saves png8 images. At 16bpc, numpngw saves png16 images. At 32 bpc, cv2 saves EXR images (and optionally tifffile saves 32bpc tiffs).
def save_8_16_or_32bpc_image(image, outdir, filename, bit_depth_output): 
    if bit_depth_output == 8: 
        image.save(os.path.join(outdir, filename))
    elif bit_depth_output == 32:
        #tifffile.imsave(os.path.join(outdir, filename).replace(".png", ".tiff"), image) # Un-comment to save 32bpc TIFF images too. Also un-comment 'import tifffile'
        cv2.imwrite(os.path.join(outdir, filename).replace(".png", ".exr"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        write_png(os.path.join(outdir, filename), image)

def next_seed(args):
    if args.seed_behavior == 'iter':
        if args.seed_internal % args.seed_iter_N == 0:
            args.seed += 1
        args.seed_internal += 1
    elif args.seed_behavior == 'ladder':
        if args.seed_internal == 0:
            args.seed += 2
            args.seed_internal = 1
        else:
            args.seed -= 1
            args.seed_internal = 0
    elif args.seed_behavior == 'alternate':
        if args.seed_internal == 0:
            args.seed += 1
            args.seed_internal = 1
        else:
            args.seed -= 1
            args.seed_internal = 0
    elif args.seed_behavior == 'fixed':
        pass # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32 - 1)
    return args.seed

def render_image_batch(root, args, cond_prompts, uncond_prompts):

    # convert prompt dict to list
    cond_prompts = [value for key, value in cond_prompts.items()]
    uncond_prompts = [value for key, value in uncond_prompts.items()]
    
    # check that the prompt lists are the same length repeat last element if not
    if len(cond_prompts) > len(uncond_prompts):
        uncond_prompts += [uncond_prompts[-1]] * (len(cond_prompts) - len(uncond_prompts))
    if len(uncond_prompts) > len(cond_prompts):
        uncond_prompts = uncond_prompts[:len(cond_prompts)]

    args.cond_prompts = {k: f"{v:05d}" for v, k in enumerate(cond_prompts)}
    args.uncond_prompts = {k: f"{v:05d}" for v, k in enumerate(uncond_prompts)}

    # set vid init to False
    args.using_vid_init = False
    
    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_settings or args.save_samples:
        print(f"Saving to {os.path.join(args.outdir, args.timestring)}_*")

    # save settings for the batch
    if args.save_settings:
        filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(dict(args.__dict__), f, ensure_ascii=False, indent=4)

    # set
    index = 0
    
    # function for init image batching
    init_array = []
    if args.use_init:
        if args.init_image == "":
            raise FileNotFoundError("No path was given for init_image")
        if args.init_image.startswith('http://') or args.init_image.startswith('https://'):
            init_array.append(args.init_image)
        elif not os.path.isfile(args.init_image):
            if args.init_image[-1] != "/": # avoids path error by adding / to end if not there
                args.init_image += "/" 
            for image in sorted(os.listdir(args.init_image)): # iterates dir and appends images to init_array
                if image.split(".")[-1] in ("png", "jpg", "jpeg"):
                    init_array.append(args.init_image + image)
        else:
            init_array.append(args.init_image)
    else:
        init_array = [""]

    # when doing large batches don't flood browser with images
    clear_between_batches = args.n_batch >= 32

    for iprompt, (cond_prompt, uncond_prompt) in enumerate(zip(cond_prompts,uncond_prompts)):
        args.cond_prompt = cond_prompt
        args.uncond_prompt = uncond_prompt
        args.clip_prompt = cond_prompt
        print(f"Prompt {iprompt+1} of {len(cond_prompts)}")
        print(f"cond_prompt: {args.cond_prompt}")
        print(f"uncond_prompt: {args.uncond_prompt}")

        all_images = []

        for batch_index in range(args.n_batch):
            if clear_between_batches and batch_index % 32 == 0: 
                display.clear_output(wait=True)            
            print(f"Batch {batch_index+1} of {args.n_batch}")
            
            for image in init_array: # iterates the init images
                args.init_image = image
                results = generate(args, root)
                for image in results:
                    if args.make_grid:
                        all_images.append(T.functional.pil_to_tensor(image))
                    if args.save_samples:
                        if args.filename_format == "{timestring}_{index}_{prompt}.png":
                            filename = f"{args.timestring}_{index:05}_{sanitize(cond_prompt)[:160]}.png"
                        else:
                            filename = f"{args.timestring}_{index:05}_{args.seed}.png"
                        save_8_16_or_32bpc_image(image, args.outdir, filename, args.bit_depth_output)
                    if args.display_samples:
                        if args.bit_depth_output != 8:
                            image = convert_image_to_8bpc(image, args.bit_depth_output)
                        display.display(image)
                    index += 1
                args.seed = next_seed(args)

        #print(len(all_images))
        if args.make_grid:
            grid = make_grid(all_images, nrow=int(len(all_images)/args.grid_rows))
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            filename = f"{args.timestring}_{iprompt:05d}_grid_{args.seed}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))
            grid_image.save(os.path.join(args.outdir, filename))
            display.clear_output(wait=True)            
            display.display(grid_image)


def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened


def render_animation(root, anim_args, args, cond_prompts, uncond_prompts):
    # handle hybrid video generation
    if anim_args.animation_mode in ['2D','3D']:
        if anim_args.hybrid_composite or anim_args.hybrid_motion in ['Affine', 'Perspective', 'Optical Flow']:
            args, anim_args, inputfiles = hybrid_generation(args, anim_args, root)
            # path required by hybrid functions, even if hybrid_comp_save_extra_frames is False
            hybrid_frame_path = os.path.join(args.outdir, 'hybridframes')

    # animations use key framed prompts
    args.cond_prompts = cond_prompts
    args.uncond_prompts = uncond_prompts

    # expand key frame strings to values
    keys = DeformAnimKeys(anim_args)

    # resume animation
    start_frame = 0
    if anim_args.resume_from_timestring:
        for tmp in os.listdir(args.outdir):
            filename = tmp.split("_")
            # don't use saved depth maps to count number of frames
            if anim_args.resume_timestring in filename and "depth" not in filename:
                start_frame += 1
        start_frame = start_frame - 1

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")

    # save settings for the batch
    settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
        json.dump(s, f, ensure_ascii=False, indent=4)
        
    # resume from timestring
    if anim_args.resume_from_timestring:
        args.timestring = anim_args.resume_timestring

    # expand cond prompts out to per-frame
    cond_prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])
    for i, prompt in cond_prompts.items():
        cond_prompt_series[int(i)] = prompt
    cond_prompt_series = cond_prompt_series.ffill().bfill()

    # expand uncond prompts out to per-frame
    uncond_prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])
    for i, prompt in uncond_prompts.items():
        uncond_prompt_series[int(i)] = prompt
    uncond_prompt_series = uncond_prompt_series.ffill().bfill()

    # check for video inits
    using_vid_init = anim_args.animation_mode == 'Video Input'
    args.using_vid_init = using_vid_init

    # load depth model for 3D (or depth compositing)
    predict_depths = (anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
    predict_depths = predict_depths or (anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type in ['Depth','Video Depth'])
    if predict_depths:
        depth_model = DepthModel(root.device)
        depth_model.load_midas(root.models_path)
        if anim_args.midas_weight < 1.0:
            depth_model.load_adabins(root.models_path)
        # depth based compositing requires saved depth maps
        if anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type =='Depth':
            anim_args.save_depth_maps = True
    else:
        depth_model = None
        anim_args.save_depth_maps = False

    # state for interpolating between diffusion steps
    turbo_steps = 1 if using_vid_init else int(anim_args.diffusion_cadence)
    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0

    # resume animation
    prev_sample = None
    color_match_sample = None
    if anim_args.resume_from_timestring and not using_vid_init:
        last_frame = start_frame-1
        if turbo_steps > 1:
            last_frame -= last_frame%turbo_steps
        path = os.path.join(args.outdir,f"{args.timestring}_{last_frame:05}.png")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prev_sample = sample_from_cv2(img)
        if anim_args.color_coherence != 'None':
            color_match_sample = img
        if turbo_steps > 1:
            turbo_next_image, turbo_next_frame_idx = sample_to_cv2(prev_sample, type=np.float32), last_frame
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            start_frame = last_frame+turbo_steps

    args.n_samples = 1
    frame_idx = start_frame
    while frame_idx < anim_args.max_frames:
        print(f"Rendering animation frame {frame_idx} of {anim_args.max_frames}")
        noise = keys.noise_schedule_series[frame_idx]
        strength = keys.strength_schedule_series[frame_idx]
        contrast = keys.contrast_schedule_series[frame_idx]
        kernel = int(keys.kernel_schedule_series[frame_idx])
        sigma = keys.sigma_schedule_series[frame_idx]
        amount = keys.amount_schedule_series[frame_idx]
        threshold = keys.threshold_schedule_series[frame_idx]
        hybrid_comp_schedules = {
            "alpha": keys.hybrid_comp_alpha_schedule_series[frame_idx],
            "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[frame_idx],
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
            "mask_auto_contrast_cutoff_low": int(keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx]),
            "mask_auto_contrast_cutoff_high": int(keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx]),
        }
        sampler_name = None
        if anim_args.enable_schedule_samplers:
            sampler_name = keys.sampler_schedule_series[frame_idx]
        depth = None
        
        # emit in-between frames
        if turbo_steps > 1:
            tween_frame_start_idx = max(0, frame_idx-turbo_steps)
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                print(f"  creating in between frame {tween_frame_idx} tween:{tween:0.2f}")

                advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                advance_next = tween_frame_idx > turbo_next_frame_idx

                if depth_model is not None:
                    assert(turbo_next_image is not None)
                    depth = depth_model.predict(turbo_next_image, anim_args)

                if advance_prev:
                    turbo_prev_image, _ = anim_frame_warp(turbo_prev_image, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device)
                if advance_next:
                    turbo_next_image, _ = anim_frame_warp(turbo_next_image, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device)

                # hybrid video motion - warps turbo_prev_image or turbo_next_image to match motion
                if tween_frame_idx > 0:
                    if anim_args.hybrid_motion in ['Affine', 'Perspective']:
                        if anim_args.hybrid_motion_use_prev_img:
                            if advance_prev:
                                matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx, (args.W, args.H), inputfiles, turbo_prev_image, anim_args.hybrid_motion)
                                turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix, anim_args.hybrid_motion, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)
                            if advance_next:
                                matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx, (args.W, args.H), inputfiles, turbo_next_image, anim_args.hybrid_motion)
                                turbo_next_image = image_transform_ransac(turbo_next_image, matrix, anim_args.hybrid_motion, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)
                        else:
                            matrix = get_matrix_for_hybrid_motion(tween_frame_idx-1, (args.W, args.H), inputfiles, anim_args.hybrid_motion)
                            if advance_prev:
                                turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix, anim_args.hybrid_motion, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)
                            if advance_next:
                                turbo_next_image = image_transform_ransac(turbo_next_image, matrix, anim_args.hybrid_motion, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)
                    if anim_args.hybrid_motion in ['Optical Flow']:
                        if anim_args.hybrid_motion_use_prev_img:
                            if advance_prev:
                                flow = get_flow_for_hybrid_motion_prev(tween_frame_idx-1, (args.W, args.H), inputfiles, hybrid_frame_path, turbo_prev_image, anim_args.hybrid_flow_method, anim_args.hybrid_comp_save_extra_frames)
                                turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)
                            if advance_next:
                                flow = get_flow_for_hybrid_motion_prev(tween_frame_idx-1, (args.W, args.H), inputfiles, hybrid_frame_path, turbo_next_image, anim_args.hybrid_flow_method, anim_args.hybrid_comp_save_extra_frames)
                                turbo_next_image = image_transform_optical_flow(turbo_next_image, flow, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)
                        else:
                            flow = get_flow_for_hybrid_motion(tween_frame_idx-1, (args.W, args.H), inputfiles, hybrid_frame_path, anim_args.hybrid_flow_method, anim_args.hybrid_comp_save_extra_frames)
                            if advance_prev:
                                turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)
                            if advance_next:
                                turbo_next_image = image_transform_optical_flow(turbo_next_image, flow, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)
                      
                # Transformed raw image before color coherence and noise. Used for mask overlay
                if args.use_mask and args.overlay_mask:
                    # Apply transforms to the original image
                    init_image_raw, _ = anim_frame_warp(args.init_sample_raw, args, anim_args, keys, frame_idx, depth_model, depth, device=root.device)
                    args.init_sample_raw = sample_from_cv2(init_image_raw).half().to(root.device)

                #Transform the mask image
                if args.use_mask:
                    if args.mask_sample is None:
                        args.mask_sample = prepare_overlay_mask(args, root, prev_sample.shape)
                    # Transform the mask
                    mask_image, _ = anim_frame_warp(args.mask_sample, args, anim_args, keys, frame_idx, depth_model, depth, device=root.device)
                    args.mask_sample = sample_from_cv2(mask_image).half().to(root.device)

                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image*(1.0-tween) + turbo_next_image*tween
                else:
                    img = turbo_next_image

                # intercept and override to grayscale
                if anim_args.color_force_grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                filename = f"{args.timestring}_{tween_frame_idx:05}.png"
                cv2.imwrite(os.path.join(args.outdir, filename), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                if anim_args.save_depth_maps:
                    depth_model.save(os.path.join(args.outdir, f"{args.timestring}_depth_{tween_frame_idx:05}.png"), depth, args.bit_depth_output)
            if turbo_next_image is not None:
                prev_sample = sample_from_cv2(turbo_next_image)

        # apply transforms to previous frame
        if prev_sample is not None:
            prev_img, depth = anim_frame_warp(prev_sample, args, anim_args, keys, frame_idx, depth_model, depth=None, device=root.device)

            # hybrid video motion - warps prev_img to match motion, usually to prepare for compositing
            if frame_idx > 0:
                if anim_args.hybrid_motion in ['Affine', 'Perspective']:
                    if anim_args.hybrid_motion_use_prev_img:
                        matrix = get_matrix_for_hybrid_motion_prev(frame_idx, (args.W, args.H), inputfiles, prev_img, anim_args.hybrid_motion)
                    else:
                        matrix = get_matrix_for_hybrid_motion(frame_idx-1, (args.W, args.H), inputfiles, anim_args.hybrid_motion)
                    prev_img = image_transform_ransac(prev_img, matrix, anim_args.hybrid_motion, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)    
                if anim_args.hybrid_motion in ['Optical Flow']:
                    if anim_args.hybrid_motion_use_prev_img:
                        flow = get_flow_for_hybrid_motion_prev(frame_idx-1, (args.W, args.H), inputfiles, hybrid_frame_path, prev_img, anim_args.hybrid_flow_method, anim_args.hybrid_comp_save_extra_frames)
                    else:
                        flow = get_flow_for_hybrid_motion(frame_idx-1, (args.W, args.H), inputfiles, hybrid_frame_path, anim_args.hybrid_flow_method, anim_args.hybrid_comp_save_extra_frames)
                    prev_img = image_transform_optical_flow(prev_img, flow, cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE)
                if anim_args.hybrid_use_video_as_mse_image:
                    args.init_mse_image = os.path.join(args.outdir, 'inputframes', f"{frame_idx:05}.jpg")
                    print(f"Using {args.init_mse_image} as init_mse_image")

            # do hybrid video - composites video frame into prev_img (now warped if using motion)
            if anim_args.hybrid_composite:
                args, prev_img = hybrid_composite(args, anim_args, frame_idx, prev_img, depth_model, hybrid_comp_schedules, root)

            # Transformed raw image before color coherence and noise. Used for mask overlay
            if args.use_mask and args.overlay_mask:
                # Apply transforms to the original image
                init_image_raw, _ = anim_frame_warp(args.init_sample_raw, args, anim_args, keys, frame_idx, depth_model, depth, device=root.device)
                args.init_sample_raw = sample_from_cv2(init_image_raw).half().to(root.device)

            #Transform the mask image
            if args.use_mask:
                if args.mask_sample is None:
                    args.mask_sample = prepare_overlay_mask(args, root, prev_sample.shape)
                # Transform the mask
                mask_sample, _ = anim_frame_warp(args.mask_sample, args, anim_args, keys, frame_idx, depth_model, depth, device=root.device)
                args.mask_sample = sample_from_cv2(mask_sample).half().to(root.device)
            
            # apply color matching
            if anim_args.color_coherence != 'None':
                # video color matching
                hybrid_available = anim_args.hybrid_composite or anim_args.hybrid_motion in ['Optical Flow', 'Affine', 'Perspective']
                if anim_args.color_coherence == 'Video Input' and hybrid_available:
                    video_color_coherence_frame = int(frame_idx) % int(anim_args.color_coherence_video_every_N_frames) == 0
                    if video_color_coherence_frame:
                        prev_vid_img = Image.open(os.path.join(args.outdir, 'inputframes', f"{frame_idx:05}.jpg"))
                        prev_vid_img = prev_vid_img.resize((args.W, args.H), Image.Resampling.LANCZOS)
                        color_match_sample = np.asarray(prev_vid_img)
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

            # intercept and override to grayscale
            if anim_args.color_force_grayscale:
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)

            # apply scaling
            contrast_sample = prev_img * contrast
            # apply anti-blur
            contrast_sample = unsharp_mask(contrast_sample, (kernel, kernel), sigma, amount, threshold)

            # apply frame noising
            noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

            # use transformed previous frame as init for current
            args.use_init = True
            args.init_sample = noised_sample.half().to(root.device)
            args.strength = max(0.0, min(1.0, strength))

        # grab prompt for current frame
        args.cond_prompt = cond_prompt_series[frame_idx]
        args.uncond_prompt = uncond_prompt_series[frame_idx]
        args.clip_prompt = cond_prompt_series[frame_idx]
        print(f"seed: {args.seed}")
        print(f"cond_prompt: {args.cond_prompt}")
        print(f"uncond_prompt: {args.uncond_prompt}")

        # assign sampler_name to args.sampler
        available_samplers = ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
        if sampler_name is not None:
            if sampler_name in available_samplers:
                args.sampler = sampler_name

        # print run info
        if not using_vid_init:
            print(f"Sampler: {args.sampler}")
            print(f"Angle: {keys.angle_series[frame_idx]} Zoom: {keys.zoom_series[frame_idx]}")
            print(f"Tx: {keys.translation_x_series[frame_idx]} Ty: {keys.translation_y_series[frame_idx]} Tz: {keys.translation_z_series[frame_idx]}")
            print(f"Rx: {keys.rotation_3d_x_series[frame_idx]} Ry: {keys.rotation_3d_y_series[frame_idx]} Rz: {keys.rotation_3d_z_series[frame_idx]}")
            print(f"noise:  {keys.noise_schedule_series[frame_idx]}")
            print(f"Strength:  {keys.strength_schedule_series[frame_idx]}")
            print(f"Contrast:  {keys.contrast_schedule_series[frame_idx]}")
            print(f"Kernel:  {int(keys.kernel_schedule_series[frame_idx])}")
            print(f"Sigma:  {keys.sigma_schedule_series[frame_idx]}")
            print(f"Amount:  {keys.amount_schedule_series[frame_idx]}")
            print(f"Threshold:  {keys.threshold_schedule_series[frame_idx]}")

        # grab init image for current frame
        if using_vid_init:
            init_frame = os.path.join(args.outdir, 'inputframes', f"{frame_idx+1:05}.jpg")            
            print(f"Using video init frame {init_frame}")
            args.init_image = init_frame
            if anim_args.use_mask_video:
                mask_frame = os.path.join(args.outdir, 'maskframes', f"{frame_idx+1:05}.jpg")
                args.mask_file = mask_frame

        # sample the diffusion model
        sample, image = generate(args, root, frame_idx, return_latent=False, return_sample=True)

        # intercept and override to grayscale
        if anim_args.color_force_grayscale:
            image = ImageOps.grayscale(image)
            image = ImageOps.colorize(image, black ="black", white ="white")

        # First image sample used for masking
        if not using_vid_init:
            prev_sample = sample
            if args.use_mask and args.overlay_mask:
                if args.init_sample_raw is None:
                        args.init_sample_raw = sample

        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = sample_to_cv2(sample, type=np.float32), frame_idx
            frame_idx += turbo_steps
        else:    
            filename = f"{args.timestring}_{frame_idx:05}.png"
            # Save image to 8bpc or 16bpc
            save_8_16_or_32bpc_image(image, args.outdir, filename, args.bit_depth_output)
            if anim_args.save_depth_maps:
                depth = depth_model.predict(sample_to_cv2(sample), anim_args)
                depth_model.save(os.path.join(args.outdir, f"{args.timestring}_depth_{frame_idx:05}.png"), depth, args.bit_depth_output)
            frame_idx += 1

        # Convert image to 8bpc to display
        if args.bit_depth_output != 8: 
            image = convert_image_to_8bpc(image, args.bit_depth_output) 

        display.clear_output(wait=True)
        display.display(image)

        args.seed = next_seed(args)

def render_input_video(args, anim_args, animation_prompts, root):
    # create a folder for the video input frames to live in
    video_in_frame_path = os.path.join(args.outdir, 'inputframes') 
    os.makedirs(video_in_frame_path, exist_ok=True)
    
    # save the video frames from input video
    print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}...")
    vid2frames(anim_args.video_init_path, video_in_frame_path, anim_args.extract_nth_frame, anim_args.overwrite_extracted_frames)

    # determine max frames from length of input frames
    anim_args.max_frames = len([f for f in pathlib.Path(video_in_frame_path).glob('*.jpg')])
    args.use_init = True
    print(f"Loading {anim_args.max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}")

    if anim_args.use_mask_video:
        # create a folder for the mask video input frames to live in
        mask_in_frame_path = os.path.join(args.outdir, 'maskframes') 
        os.makedirs(mask_in_frame_path, exist_ok=True)

        # save the video frames from mask video
        print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {mask_in_frame_path}...")
        vid2frames(anim_args.video_mask_path, mask_in_frame_path, anim_args.extract_nth_frame, anim_args.overwrite_extracted_frames)
        args.use_mask = True
        args.overlay_mask = True

    render_animation(args, anim_args, animation_prompts, root)

def render_interpolation(args, anim_args, animation_prompts, root):
    # animations use key framed prompts
    args.prompts = animation_prompts

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")

    # save settings for the batch
    settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
        json.dump(s, f, ensure_ascii=False, indent=4)
    
    # Interpolation Settings
    args.n_samples = 1
    args.seed_behavior = 'fixed' # force fix seed at the moment bc only 1 seed is available
    prompts_c_s = [] # cache all the text embeddings

    print(f"Preparing for interpolation of the following...")

    for i, prompt in animation_prompts.items():
        args.prompt = prompt
        args.clip_prompt = args.prompt

        # sample the diffusion model
        results = generate(args, root, return_c=True)
        c, image = results[0], results[1]
        prompts_c_s.append(c) 
      
        # Convert image to 8bpc to display
        if args.bit_depth_output != 8: 
            image = convert_image_to_8bpc(image, args.bit_depth_output) 
      
        # display.clear_output(wait=True)
        display.display(image)
      
        args.seed = next_seed(args)

    display.clear_output(wait=True)
    print(f"Interpolation start...")

    frame_idx = 0

    if anim_args.interpolate_key_frames:
        for i in range(len(prompts_c_s)-1):
            dist_frames = list(animation_prompts.items())[i+1][0] - list(animation_prompts.items())[i][0]
            if dist_frames <= 0:
                print("key frames duplicated or reversed. interpolation skipped.")
                return
        else:
            for j in range(dist_frames):
                # interpolate the text embedding
                prompt1_c = prompts_c_s[i]
                prompt2_c = prompts_c_s[i+1]  
                args.init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1/dist_frames))

                # sample the diffusion model
                results = generate(args, root)
                image = results[0]

                filename = f"{args.timestring}_{frame_idx:05}.png"
                # Save image to 8bpc or 16bpc
                save_8_16_or_32bpc_image(image, args.outdir, filename, args.bit_depth_output)
                frame_idx += 1

                # Convert image to 8bpc to display
                if args.bit_depth_output != 8: 
                    image = convert_image_to_8bpc(image, args.bit_depth_output) 

                display.clear_output(wait=True)
                display.display(image)

                args.seed = next_seed(args)

    else:
        for i in range(len(prompts_c_s)-1):
            for j in range(anim_args.interpolate_x_frames+1):
                # interpolate the text embedding
                prompt1_c = prompts_c_s[i]
                prompt2_c = prompts_c_s[i+1]  
                args.init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1/(anim_args.interpolate_x_frames+1)))

                # sample the diffusion model
                results = generate(args, root)
                image = results[0]

                filename = f"{args.timestring}_{frame_idx:05}.png"
                save_8_16_or_32bpc_image(image, args.outdir, filename, args.bit_depth_output)
                frame_idx += 1

                # Convert image to 8bpc to display
                if args.bit_depth_output != 8: 
                    image = convert_image_to_8bpc(image, args.bit_depth_output) 

                display.clear_output(wait=True)
                display.display(image)

                args.seed = next_seed(args)

    # generate the last prompt
    args.init_c = prompts_c_s[-1]
    results = generate(args, root)
    image = results[0]
    filename = f"{args.timestring}_{frame_idx:05}.png"
    save_8_16_or_32bpc_image(image, args.outdir, filename, args.bit_depth_output)

    # Convert image to 8bpc to display
    if args.bit_depth_output != 8: 
        image = convert_image_to_8bpc(image, args.bit_depth_output) 

    display.clear_output(wait=True)
    display.display(image)
    args.seed = next_seed(args)

    #clear init_c
    args.init_c = None
