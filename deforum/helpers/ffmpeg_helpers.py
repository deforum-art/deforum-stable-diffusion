import os
import subprocess
from base64 import b64encode
from IPython import display
import cv2

def get_extension_maxframes(args,ffmpeg_outdir,ffmpeg_timestring):
    all_frames = []
    for file in os.listdir(args.outdir):
        if file.startswith(args.timestring):
            if file.endswith(('.png', '.jpg', '.jpeg', '.exr')):
                all_frames.append(file)
    extension = all_frames[0].split(".")[-1]
    max_frames = len(all_frames)
    return extension, max_frames

def get_auto_outdir_timestring(args,ffmpeg_mode):
    try:
        ffmpeg_outdir = args.outdir
        ffmpeg_timestring = args.timestring
        return ffmpeg_outdir, ffmpeg_timestring
    except Exception as e:
        print(e)
        print("ffmpeg mode set to auto and args not defined in global scope")

def get_ffmpeg_path(ffmpeg_outdir, ffmpeg_timestring, ffmpeg_extension):
    try:
        ffmpeg_image_path = os.path.join(ffmpeg_outdir,ffmpeg_timestring+f"_%05d.{ffmpeg_extension}")
        ffmpeg_mp4_path = os.path.join(ffmpeg_outdir,ffmpeg_timestring+".mp4")
        ffmpeg_gif_path = os.path.join(ffmpeg_outdir,ffmpeg_timestring+".gif")
        return ffmpeg_image_path, ffmpeg_mp4_path, ffmpeg_gif_path
    except Exception as e:
        print(e)
        print("error making path variables redefine args or use different mode")

def make_mp4_ffmpeg(ffmpeg_args, display_ffmpeg=False, debug=False):

    if os.path.exists("./ffmpeg"):
        ffmpeg_path = './ffmpeg'
    else:
        ffmpeg_path = 'ffmpeg'

    # make mp4
    cmd = [
        ffmpeg_path,
        '-y',
        '-vcodec', "png",
        '-r', str(ffmpeg_args.ffmpeg_fps),
        '-start_number', str(0),
        '-i', ffmpeg_args.ffmpeg_image_path,
        '-frames:v', str(ffmpeg_args.ffmpeg_maxframes),
        '-c:v', 'libx264',
        '-vf',
        f'fps={ffmpeg_args.ffmpeg_fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        '-pattern_type', 'sequence',
        ffmpeg_args.ffmpeg_mp4_path
    ]

    if debug:
        print(cmd)

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)

    if display_ffmpeg == True:
        mp4 = open(ffmpeg_args.ffmpeg_mp4_path,'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        display.display(display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>'))


def make_gif_ffmpeg(ffmpeg_args, debug=False):

    if os.path.exists("./ffmpeg"):
        ffmpeg_path = './ffmpeg'
    else:
        ffmpeg_path = 'ffmpeg'

    # make gif
    cmd = [
        ffmpeg_path,
        '-y',
        '-vcodec', "png",
        '-r', str(ffmpeg_args.ffmpeg_fps),
        '-start_number', str(0),
        '-i', ffmpeg_args.ffmpeg_image_path,
        '-frames:v', str(ffmpeg_args.ffmpeg_maxframes),
        '-vf',
        f'fps={ffmpeg_args.ffmpeg_fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        '-pattern_type', 'sequence',
        ffmpeg_args.ffmpeg_gif_path
    ]

    if debug:
        print(cmd)

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)

def patrol_cycle(args,ffmpeg_args, debug=False):
        cap = cv2.VideoCapture(ffmpeg_args.ffmpeg_mp4_path)
        # Get the video frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # Reverse the frames
        reversed_frames = frames[::-1]

        # Append the original frames to the reversed frames
        frames = frames + reversed_frames
        # Create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(ffmpeg_args.ffmpeg_mp4_path, fourcc, float(ffmpeg_args.ffmpeg_fps), (args.W, args.H), isColor=True)
        # Write the frames to the output video
        for frame in frames:
            out.write(frame)
        # Release the video writer and capture objects
        out.release()
        cap.release()