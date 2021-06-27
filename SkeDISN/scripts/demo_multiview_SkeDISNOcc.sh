gpu=0
predske_dir='{specify your predicted skeleton volume directory}'
view_num=3
img_path='03001627_2c250a89e731a3d16f554fd9e81f2ffc_12_13_19'
#img_path='04379243_c3c467718eb9b2a313f96345312df593_11_16_17'
log_dir='checkpoint/skedisn_occ'
restore_modeldisn='checkpoint/disn_occ/model.ckpt'
python test/create_sdf_multiview_add_skevox.py --ske_local_patch_share --binary  --img_feat_twostream \
    --sdf_res 256 --gpu $gpu --log_dir $log_dir --category 'all' --restore_modeldisn $restore_modeldisn \
    --use_predvox --predvox_dir  $predske_dir  --img_path $img_path --view_num $view_num
