export CUDA_VISIBLE_DEVICES=0
batchsize=16 ###
nepoch=20
cat='all'
model_ske='checkpoints/all/SVR_CurSur_all_20/network.pth'
model_vol='checkpoints/all/Im2Ske_all_Local/network.pth'
env='Im2Ske_all_End2end'
lr1=1e-4
lr2=1e-3
weight_pts=1.0
weight_vox=0.1
lrStep=10
start_eval_epoch=0
patch_num=8
wce=1.5
python volume_train/train_end2end.py --batchSize $batchsize --nepoch $nepoch --category $cat \
    --load_sperate  --model_ske $model_ske  --model_vol $model_vol --env $env --super_points 2500 --samples_line 4000 --samples_triangle 24000 \
    --lr1 $lr1 --lr2 $lr2 --lrStep $lrStep --patch_num $patch_num --start_eval_epoch $start_eval_epoch --woptfeat --weight_ce $wce --weight_pts $weight_pts --weight_vox $weight_vox
