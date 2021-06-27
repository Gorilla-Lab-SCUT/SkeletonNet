gpu=0
basemesh_root='your own basemesh info folder'
checkpoint='all_e2000_n1'

CUDA_VISIBLE_DEVICES=$gpu python gen.py --basemesh_root $basemesh_root --checkpoint $checkpoint --vertex_chamfer True
