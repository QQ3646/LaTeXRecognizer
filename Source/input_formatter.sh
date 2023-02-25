#!/bin/bash
mkdir ./formatted_input
for f in $(find ./ -name '*.jpg');
do
echo "Processing $f"
file=${f##*/}
echo "a.k.a. $file"
ffmpeg -i "$f" -vf "[in] scale='min(512,iw)':-1:force_original_aspect_ratio=increase,pad=512:512:-1:-1:color=white, colorkey=white [out]" "./formatted_input/${file%.jpg}.png"
done