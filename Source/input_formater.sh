#!/bin/bash
mkdir ./formatted_input
for f in $(find ./ -name '*.jpg');
do
echo "Processing $f"
file=${f##*/}
echo "a.k.a. $file"
ffmpeg -i "$f" -vf "[in] scale=512:-1:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2,colorkey=black [out]" "./formatted_input/${file%.jpg}.png"
done