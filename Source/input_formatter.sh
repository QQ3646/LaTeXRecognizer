#!/bin/bash
for f in $(find ../input/ -name '*.png');
do
echo "Processing $f"
file=${f##*/}
echo "a.k.a. $file"
dirpath=${f%/*/*}
mkdir -p ${dirpath}/formatted_input
ffmpeg -i "$f" -vf "[in] scale='min(515,iw)':-1:force_original_aspect_ratio=decrease,pad=515:515:-1:-1:color=white, colorkey=white [out]" "${dirpath}/formatted_input/${file%.png}.png"
done