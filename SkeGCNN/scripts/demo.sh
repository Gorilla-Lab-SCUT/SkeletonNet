gpu=0
basemesh_root='../demo_results'
checkpoint='./checkpoint/skegcnn'
cat='03001627'
mod='1e304b967d5253d5dd079f8cece51712'
idx=0

CUDA_VISIBLE_DEVICES=$gpu python demo.py --basemesh_root $basemesh_root --checkpoint $checkpoint --catname $cat --modname $mod --index $idx
