from skimage.exposure import match_histograms
import cv2
import numpy as np

def maintain_colors(prev_img, color_match_sample, mode):
    print(f"Mode: {mode}")
    print(f"Input image shape: {prev_img.shape}, dtype: {prev_img.dtype}")
    print(f"Color match sample shape: {color_match_sample.shape}, dtype: {color_match_sample.dtype}")
    
    if mode == 'Match Frame 0 RGB':
        print("RGB values before matching:")
        print(f"prev_img min: {prev_img.min()}, max: {prev_img.max()}")
        print(f"color_match_sample min: {color_match_sample.min()}, max: {color_match_sample.max()}")
        
        matched = match_histograms(prev_img, color_match_sample, channel_axis=-1)
        
        print("RGB values after matching:")
        print(f"matched min: {matched.min()}, max: {matched.max()}")
        
        return (matched * 255).astype(np.uint8)
    
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        
        print("HSV values before matching:")
        print(f"prev_img_hsv min: {prev_img_hsv.min()}, max: {prev_img_hsv.max()}")
        print(f"color_match_hsv min: {color_match_hsv.min()}, max: {color_match_hsv.max()}")
        
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, channel_axis=-1)
        
        print("HSV values after matching:")
        print(f"matched_hsv min: {matched_hsv.min()}, max: {matched_hsv.max()}")
        
        matched_hsv = (matched_hsv * 255).astype(np.uint8)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    
    else:  # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        
        print("LAB values before matching:")
        print(f"prev_img_lab min: {prev_img_lab.min()}, max: {prev_img_lab.max()}")
        print(f"color_match_lab min: {color_match_lab.min()}, max: {color_match_lab.max()}")
        
        matched_lab = match_histograms(prev_img_lab, color_match_lab, channel_axis=-1)
        
        print("LAB values after matching:")
        print(f"matched_lab min: {matched_lab.min()}, max: {matched_lab.max()}")
        
        matched_lab = (matched_lab * 255).astype(np.uint8)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
