#!/bin/bash

if [ $# -ne 1 ]; then
  echo "One arguments are required." 1>&2
  echo "sh demo_img.sh PATH_TO_YOUR_INPUT_FOLDER"
  exit 1
fi
video_name="$(basename $1)"

python3 infer_video.py -i $1 -o ./demo/infer_video -m ./trained_models/model_1st.pth --gpu 0
light_name="cluster32"
python3 relighting_video_1st.py -i ./demo/infer_video/${video_name} -o ./demo/relighting_video/1st -l ./data/test_light/${light_name}.npy
python3 relighting_video_2nd.py -i ./demo/relighting_video/1st/${video_name}+${light_name} -o ./demo/relighting_video/2nd -m ./trained_models/model_2nd.pth --gpu 0
python3 remove_flicker.py -i $1 -o ./demo/relighting_video/flicker_reduction -p ./demo/relighting_video/2nd/${video_name}+${light_name} -l ./demo/relighting_video/1st/${video_name}+${light_name} --gpu 0


cat <<__EOT__

FINISH!
__EOT__
exit 0