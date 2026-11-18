#!/bin/bash
# Encode the rendered frames (frames7/f_%05d.png, 720 x 1280) to 1080 x 1920 H.264 at 60 fps with the edited song.
# song5.wav (the song edit the timing is built on) is a commercial recording and is not included.
set -e
cd "$(dirname "$0")"
mkdir -p out enc
OUT=out/Herobrine_VS_Notch_Edit_v7_1080x1920_60fps.mp4
VF="scale=1080:1920:flags=lanczos,unsharp=5:5:0.4:5:5:0,format=yuv420p"
ffmpeg -y -v error -framerate 60 -i frames7/f_%05d.png -vf "$VF" -c:v libx264 -preset slow -b:v 16M -maxrate 22M -bufsize 30M -pass 1 -passlogfile enc/v7pass -an -f mp4 /dev/null
ffmpeg -y -v error -framerate 60 -i frames7/f_%05d.png -i song5.wav \
  -filter_complex "[0:v]$VF[v];[1:a]atrim=0:34.318,asetpts=PTS-STARTPTS,loudnorm=I=-14:TP=-1.0:LRA=11,aresample=48000,afade=t=out:st=34.008:d=0.31[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -preset slow -b:v 16M -maxrate 22M -bufsize 30M -pass 2 -passlogfile enc/v7pass \
  -c:a aac -b:a 256k -ar 48000 -t 34.318 -movflags +faststart -r 60 "$OUT"
ls -la "$OUT"; ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height -of default=nw=1 "$OUT"
echo ENC_DONE
