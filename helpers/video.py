import os
import pathlib
import subprocess
from base64 import b64encode
from IPython import display

def frames2vid(args, anim_args, basics):
    print(f"{basics.image_path} -> {basics.mp4_path}")

    if not basics.use_manual_settings:
        if basics.render_steps: # render steps from a single image
            fname = f"{basics.path_name_modifier}_%05d.png"
            all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir,d))]
            newest_dir = max(all_step_dirs, key=os.path.getmtime)
            basics.image_path = os.path.join(newest_dir, fname)
            print(f"Reading images from {basics.image_path}")
            basics.mp4_path = os.path.join(newest_dir, f"{args.timestring}_{basics.path_name_modifier}.mp4")
            basics.max_frames = str(args.steps)
        else: # render images for a video
            basics.image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.{basics.bitdepth_extension}")
            basics.mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
            basics.max_frames = str(anim_args.max_frames)

    # make video
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', basics.bitdepth_extension,
        '-r', str(basics.fps),
        '-start_number', str(0),
        '-i', basics.image_path,
        '-frames:v', basics.max_frames,
        '-c:v', 'libx264',
        '-vf',
        f'fps={basics.fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        '-pattern_type', 'sequence',
        basics.mp4_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)

    mp4 = open(basics.mp4_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display.display(display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )
    
    if basics.make_gif:
         gif_path = os.path.splitext(basics.mp4_path)[0]+'.gif'
         cmd_gif = [
             'ffmpeg',
             '-y',
             '-i', basics.mp4_path,
             '-r', str(basics.fps),
             gif_path
         ]
         process_gif = subprocess.Popen(cmd_gif, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def vid2frames(video_path, frames_path, n=1, overwrite=True):      
    if not os.path.exists(frames_path) or overwrite: 
        try:
            for f in pathlib.Path(video_in_frame_path).glob('*.jpg'):
            	f.unlink()
        except:
            pass
        assert os.path.exists(video_path), f"Video input {video_path} does not exist"
          
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        t=1
        success = True
        while success:
            if count % n == 0:
                cv2.imwrite(frames_path + os.path.sep + f"{t:05}.jpg" , image)     # save frame as JPEG file
                t += 1
        success,image = vidcap.read()
        count += 1
        print("Converted %d frames" % count)
    else: print("Frames already unpacked")