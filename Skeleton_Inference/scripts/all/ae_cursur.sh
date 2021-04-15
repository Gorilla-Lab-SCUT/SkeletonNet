gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
num_prim=20

cats='all'
nepoch=150
num_points_line=600
nb_primitives_line=$num_prim #10 20 30 40
num_points_square=2000
nb_primitives_square=$num_prim #10 20 30 40
super_points=2500
env='AE_all_CurSur_'$num_prim
k1=0.2
lr=1e-3
eval_epoch=100

python skeleton_train/AE_CurSur.py --nepoch $nepoch \
    --num_points_line $num_points_line --nb_primitives_line $nb_primitives_line \
    --num_points_square $num_points_square --nb_primitives_square $nb_primitives_square \
    --super_points $super_points --env $env --k1 $k1 --lr $lr --category $cats --start_eval_epoch $eval_epoch \