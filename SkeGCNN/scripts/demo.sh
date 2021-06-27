gpu=0
basemesh_root='./demo/demo_allcats_basemesh'
checkpoint='all_e2000_n1'
cat='04530566'
mod='6313352481b445a8ecbbed03ea2b4cdc'
idx=00

CUDA_VISIBLE_DEVICES=$gpu python demo.py --basemesh_root $basemesh_root --checkpoint $checkpoint --catname $cat --modname $mod --index $idx
