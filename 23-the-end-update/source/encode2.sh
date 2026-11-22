#!/bin/bash
# frames/f_%05d.jpg (1080 x 1920, 60 fps) + audio_edit.wav (from end_audio.py) -> H.264 / AAC
set -e
cd "$(dirname "$0")"
OUT=out/The_End_Update_1080x1920_60fps.mp4
ffmpeg -y -v error -framerate 60 -i frames/f_%05d.jpg -vf "format=yuv420p" -c:v libx264 -preset slow -b:v 18M -maxrate 24M -bufsize 32M -pass 1 -passlogfile endpass -an -f mp4 /dev/null
ffmpeg -y -v error -framerate 60 -i frames/f_%05d.jpg -i audio_edit.wav \
  -filter_complex "[0:v]format=yuv420p[v];[1:a]atrim=0:30.917,asetpts=PTS-STARTPTS,loudnorm=I=-14:TP=-1.0:LRA=11,aresample=48000,afade=t=out:st=30.5:d=0.4[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -preset slow -b:v 18M -maxrate 24M -bufsize 32M -pass 2 -passlogfile endpass \
  -c:a aac -b:a 256k -ar 48000 -t 30.917 -movflags +faststart -r 60 "$OUT"
ls -la "$OUT"; ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height -of default=nw=1 "$OUT"
echo ENC_DONE
