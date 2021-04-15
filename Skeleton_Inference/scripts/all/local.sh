export CUDA_VISIBLE_DEVICES=0
batchsize=4
nepoch=10
cats='all'
model_ske='checkpoints/'$cats'/SVR_CurSur_all_20/network.pth'
model_glo='checkpoints/'$cats'/Im2Ske_all_Global/network.pth'
env='Im2Ske_all_Local'
lr=1e-4
patch_num=32
eval_epoch=0
wce=1.5
python volume_train/train_local.py --batchSize $batchsize --nepoch $nepoch  --category $cats --model_ske $model_ske --model_glo $model_glo --env $env --lr $lr  --patch_num $patch_num --start_val_epoch $eval_epoch --weight_ce $wce