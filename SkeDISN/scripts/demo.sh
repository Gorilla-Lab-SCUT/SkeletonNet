gpu=0
predske_dir='../demo_results'
img_path='03001627_1e304b967d5253d5dd079f8cece51712_00'
log_dir='checkpoint/skedisn_occ'
python test/create_sdf_add_skevox.py --ske_local_patch_share  --binary  --img_feat_twostream \
    --sdf_res 256 --gpu $gpu --log_dir $log_dir --category 'all' \
    --use_predvox --predvox_dir  $predske_dir --img_path $img_path
