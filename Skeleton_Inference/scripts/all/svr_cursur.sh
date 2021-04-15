gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
num_prim=20

nepoch=150
cats='all'
model_ae='checkpoints/all/AE_all_CurSur_20/network.pth'
num_points_line=600
nb_primitives_line=$num_prim #10 20 30 40
num_points_square=2000
nb_primitives_square=$num_prim #10 20 30 40
super_points=2500
env='SVR_CurSur_all_'$num_prim
fix_decoder='True'  #only train img encoder
k1=0.2
lr=1e-3
start_eval_epoch=100

python skeleton_train/SVR_CurSur.py --nepoch $nepoch --model_ae $model_ae \
    --num_points_line $num_points_line --nb_primitives_line $nb_primitives_line \
    --num_points_square $num_points_square --nb_primitives_square $nb_primitives_square \
    --super_points $super_points --env $env --fix_decoder $fix_decoder \
    --k1 $k1 --lr $lr --start_eval_epoch $start_eval_epoch  --category $cats
