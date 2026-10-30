"""Make a locally stored side-by-side clip from TWO real user-provided preview videos.
No inference is rerun and no output is uploaded. The source recording, camera order,
time basis, and experimental status must be explained manually in the showcase page.

python scripts/build_showcase_video.py --baseline outputs/old/preview.mp4 \
 --experimental outputs/new/preview.mp4 --output outputs/showcase/side_by_side.mp4
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess


def build(before, after, output, seconds=30, fps=5, ffmpeg='ffmpeg'):
    before,after,output=map(Path,(before,after,output))
    if not before.is_file():raise FileNotFoundError(before)
    if not after.is_file():raise FileNotFoundError(after)
    if before.resolve()==after.resolve():raise ValueError('Select two different files')
    if output.exists():raise FileExistsError('Refusing to overwrite existing output')
    if seconds<=0 or fps<=0:raise ValueError('seconds and fps must be positive')
    if not shutil.which(ffmpeg):raise RuntimeError('ffmpeg executable not found on PATH')
    output.parent.mkdir(parents=True,exist_ok=True)
    # The constant 5 FPS is video playback, not the measured processing FPS.
    # Trim both inputs to the same presentation period; stop if either is shorter.
    layout=f"fps={fps},trim=duration={seconds},setpts=PTS-STARTPTS,scale=800:450:force_original_aspect_ratio=decrease,pad=800:450:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    complex_filter=(f'[0:v]{layout},drawtext=text=BASELINE:fontcolor=white:fontsize=26:'
        'x=12:y=12:box=1:boxcolor=black@0.6[left];'
        f'[1:v]{layout},drawtext=text=EXPERIMENTAL:fontcolor=white:fontsize=26:'
        'x=12:y=12:box=1:boxcolor=black@0.6[right];'
        '[left][right]hstack=inputs=2:shortest=1,format=yuv420p[out]')
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-i',str(before),
         '-i',str(after),'-filter_complex',complex_filter,'-map','[out]','-an',
         '-c:v','libx264','-preset','veryfast','-crf','24','-pix_fmt','yuv420p',
         '-movflags','+faststart',str(output)]
    result=subprocess.run(cmd,capture_output=True,text=True,check=False)
    if result.returncode!=0:
        output.unlink(missing_ok=True)
        raise RuntimeError('FFmpeg failed. Requires libx264 and drawtext filters. '+result.stderr[-1200:])
    print('Created:',output)
    print('This is a recorded side-by-side visualization, not simultaneous live inference.')
    print('Before publishing, check permission to redistribute the source footage.')
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--experimental',required=True)
    ap.add_argument('--output',default='outputs/showcase/side_by_side.mp4')
    ap.add_argument('--seconds',type=int,default=30)
    ap.add_argument('--fps',type=int,default=5)
    a=ap.parse_args()
    build(a.baseline,a.experimental,a.output,a.seconds,a.fps)
if __name__=='__main__':main()
