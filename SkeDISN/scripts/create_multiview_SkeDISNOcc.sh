gpu=0
sdf_res=64
view_num=3
log_dir='checkpoint/skedisn_occ'
disnmodel='checkpoint/disn_occ/model.ckpt --category all'
predvox_dir='{specify your predicted skeleton volume directory}'

python -u test/create_sdf_add_skevox.py --ske_local_patch_share --binary --img_feat_twostream --sdf_res $sdf_res --gpu $gpu \
    --log_dir $log_dir --restore_modeldisn $disnmodel --use_predvox --predvox_dir $predvox_dir --view_num $view_num --test_allset 