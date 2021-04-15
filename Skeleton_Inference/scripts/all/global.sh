export CUDA_VISIBLE_DEVICES=2
batchsize=16
nepoch=20
cats='all'
model_ske='checkpoints/'$cats'/SVR_CurSur_all_20/network.pth'
env='Im2Ske_all_Global'
lr=1e-4
python volume_train/train_global.py --batchSize $batchsize --nepoch $nepoch  --category $cats --model_ske $model_ske --env $env --lr $lr
