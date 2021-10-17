#!/bin/bash

if [ $# -ne 1 ]; then
  echo "One arguments are required." 1>&2
  echo "sh demo_img.sh PATH_TO_YOUR_INPUT_FOLDER"
  exit 1
fi

python3 infer_image.py -i $1 -o ./demo/infer_image -m ./trained_models/model_1st.pth --gpu 1
light_name="cluster32"
python3 relighting_image_1st.py -i ./demo/infer_image -o ./demo/relighting_image/1st -l ./data/test_light/"$light_name".npy
python3 relighting_image_2nd.py -i ./demo/relighting_image/1st -o ./demo/relighting_image/2nd -m ./trained_models/model_2nd.pth --gpu 1

cat <<__EOT__
FINISH!
__EOT__
exit 0