import numpy as np
import cv2
import random
import math
import os
import sys
import scipy.io as sio
import torch
from scipy import stats

def define_lines(num_points=600, nb_primitives=20):
    grain = int(num_points / nb_primitives) - 1
    grain = grain * 1.0
    print(grain)

    vertices = []
    for i in range(0, int(grain + 1)):
        vertices.append([i / grain])
    grid_list = [vertices for i in range(0, nb_primitives)]

    lines = []
    for prim in range(0, nb_primitives):
        for i in range(0,int(grain)):
            lines.append([(grain+1)*prim +i, (grain+1)*prim + (i+1)])
    lines_array = np.array(lines)
    lines_array = lines_array.astype(int)

    lines_adjacent = []
    lines_adjacent.append([1, 1])
    for i in range(1, int(grain)):
        lines_adjacent.append([i - 1, i + 1])
    lines_adjacent.append([int(grain - 1), int(grain - 1)])
    lines_adjacent_tensor = torch.cuda.LongTensor(lines_adjacent)
    return grid_list, lines_array, lines_adjacent_tensor

def define_squares(num_points=2000, nb_primitives=20):
    grain = int(np.sqrt(num_points / nb_primitives)) - 1
    grain = grain * 1.0
    print(grain)

    vertices = []
    for i in range(0, int(grain + 1)):
        for j in range(0, int(grain + 1)):
            vertices.append([i / grain, j / grain])
    grid_list = [vertices for i in range(0, nb_primitives)]

    faces = []
    for prim in range(0, nb_primitives):
        for i in range(1, int(grain + 1)):
            for j in range(0, int(grain + 1) - 1):
                faces.append([(grain + 1) * (grain + 1) * prim + j + (grain + 1) * i,
                              (grain + 1) * (grain + 1) * prim + j + (grain + 1) * i + 1,
                              (grain + 1) * (grain + 1) * prim + j + (grain + 1) * (i - 1)])
        for i in range(0, int(grain + 1) - 1):
            for j in range(1, int(grain + 1)):
                faces.append([(grain + 1) * (grain + 1) * prim + j + (grain + 1) * i,
                              (grain + 1) * (grain + 1) * prim + j + (grain + 1) * i - 1,
                              (grain + 1) * (grain + 1) * prim + j + (grain + 1) * (i + 1)])
    faces_array = np.array(faces)
    faces_array = faces_array.astype(int)

    edge = []
    for i, j in enumerate(faces_array):
        edge.append(j[:2])
        edge.append(j[1:])
        edge.append(j[[0, 2]])
    edge = np.array(edge)
    edge_im = edge[:, 0] * edge[:, 1] + (edge[:, 0] + edge[:, 1]) * 1j
    unique = np.unique(edge_im, return_index=True)[1]
    edge_unique = edge[unique]

    vertex_adj_matrix=[]
    for i in range(0, len(vertices)):
        vertex_adj=edge_unique[np.where((edge_unique==i).astype(int).sum(axis=1)==1)[0]].reshape(-1)
        i_index=np.where(vertex_adj!=i)[0]
        vertex_adj=vertex_adj[i_index]
        vertex_adj=vertex_adj.repeat(12/len(vertex_adj))
        vertex_adj_matrix.append(vertex_adj)
    vertex_adj_matrix = np.array(vertex_adj_matrix)
    vertex_adj_matrix_tensor = torch.from_numpy(vertex_adj_matrix).type(torch.cuda.LongTensor)
    return grid_list, faces_array, vertex_adj_matrix_tensor

def curve_laplacian(pointsReconstructed, nb_primitives, lines_adjacent_tensor):
    pointsReconstructed_line = pointsReconstructed.view(pointsReconstructed.size()[0] * nb_primitives, \
        pointsReconstructed.size()[1] // nb_primitives, pointsReconstructed.size()[2])
    vertex_adjacent_coor = torch.index_select(pointsReconstructed_line, 1, lines_adjacent_tensor.view(-1)). \
        view(pointsReconstructed_line.size()[0], lines_adjacent_tensor.size()[0], 
        lines_adjacent_tensor.size()[1], pointsReconstructed_line.size()[2])
    vertex_adjacent_coor_mean = torch.mean(vertex_adjacent_coor, 2).squeeze(2)
    laplacian_smooth = torch.abs(pointsReconstructed_line - vertex_adjacent_coor_mean)
    laplacian_smooth = laplacian_smooth.view(laplacian_smooth.size()[0] // nb_primitives,
        laplacian_smooth.size()[1] * nb_primitives, laplacian_smooth.size()[2])
    return laplacian_smooth

def surface_laplacian(pointsReconstructed, nb_primitives, vertex_adj_matrix_tensor):
    pointsReconstructed_square = pointsReconstructed.view(pointsReconstructed.size()[0] * nb_primitives,
        pointsReconstructed.size()[1] // nb_primitives, pointsReconstructed.size()[2])
    vertex_adj_coor = torch.index_select(pointsReconstructed_square, 1, vertex_adj_matrix_tensor.view(-1)).view(
        pointsReconstructed_square.size()[0], vertex_adj_matrix_tensor.size()[0], vertex_adj_matrix_tensor.size()[1],
        pointsReconstructed_square.size()[2])
    vertex_adj_coor_mean = torch.mean(vertex_adj_coor, 2).squeeze(2)
    laplacian_smooth = torch.abs(pointsReconstructed_square - vertex_adj_coor_mean)
    laplacian_smooth = laplacian_smooth.view(laplacian_smooth.size()[0] // nb_primitives,
        laplacian_smooth.size()[1] * nb_primitives, laplacian_smooth.size()[2])
    return laplacian_smooth

# how I sample from the lines
# verts = list of vertex positions : Nx3
# lines = list of lines, as vertex indecies in verts 
# feats = features of verts : CxN
# num = number of points to sample 
def sample_lines_feats(verts, lines_array, feats, num=2000): 
    lines = torch.tensor(lines_array, dtype=torch.int64).cuda()
    dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(feats.device), torch.tensor([1.0]).cuda(feats.device))
    # calculate length of each line
    x1,x2,x3 = torch.split(torch.index_select(verts, 0, lines[:,0]) - torch.index_select(verts, 0, lines[:,1]), 1, dim = 1)
    Lengths = torch.sqrt(x1**2 + x2**2 + x3**2)
    Lengths = Lengths / torch.sum(Lengths) # percentage of each line w.r.t. full curve lines
    # define descrete distribution w.r.t. line area ratios caluclated 
    choices = np.arange(Lengths.shape[0])
    dist = stats.rv_discrete(name='custm', values=(choices, Lengths.data.cpu().numpy()))
    choices = dist.rvs(size=num) # list of lines to be sampled from 

    # from each face sample a point 
    select_lines = lines[choices]
    xs = torch.index_select(verts, 0, select_lines[:,0])
    ys = torch.index_select(verts, 0, select_lines[:,1])
    xfeat = torch.index_select(feats, 0, select_lines[:, 0])
    yfeat = torch.index_select(feats, 0, select_lines[:, 1])
    u = dist_uni.sample_n(num)
    points = (1- u)*xs + u*ys
    sample_feats = (1-u)*xfeat + u*yfeat
    return points, sample_feats


def sample_triangles_feats(verts, faces_array, feats, num=10000): 
    faces = torch.tensor(faces_array, dtype=torch.int64).cuda()
    dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(feats.device), torch.tensor([1.0]).cuda(feats.device))
    # calculate area of each face 
    x1,x2,x3 = torch.split(torch.index_select(verts, 0,faces[:,0]) - torch.index_select(verts, 0,faces[:,1]), 1, dim = 1)
    y1,y2,y3 = torch.split(torch.index_select(verts, 0,faces[:,1]) - torch.index_select(verts, 0,faces[:,2]), 1, dim = 1)
    a = (x2*y3-x3*y2)**2
    b = (x3*y1 - x1*y3)**2
    c = (x1*y2 - x2*y1)**2
    Areas = torch.sqrt(a+b+c)/2
    Areas = Areas / torch.sum(Areas) # percentage of each face w.r.t. full surface area 
    # define descrete distribution w.r.t. face area ratios caluclated 
    choices = np.arange(Areas.shape[0])
    dist = stats.rv_discrete(name='custm', values=(choices, Areas.data.cpu().numpy()))
    choices = dist.rvs(size=num)# list of faces to be sampled from

    # from each face sample a point
    select_faces = faces[choices]
    xs = torch.index_select(verts, 0, select_faces[:,0])
    ys = torch.index_select(verts, 0, select_faces[:,1])
    zs = torch.index_select(verts, 0, select_faces[:,2])
    xfeat = torch.index_select(feats, 0, select_faces[:, 0])
    yfeat = torch.index_select(feats, 0, select_faces[:, 1])
    zfeat = torch.index_select(feats, 0, select_faces[:, 2])
    u = torch.sqrt(dist_uni.sample_n(num))
    v = dist_uni.sample_n(num)
    points = (1- u)*xs + (u*(1-v))*ys + u*v*zs
    sample_feats = (1-u)*xfeat + (u*(1-v)*yfeat) + u*v*zfeat
    return points, sample_feats

def batch_sample_lines_feats(curves, lines_array, curve_feat, num):
    curve_samples = []
    curve_sample_feats = []
    for j in range(0, curves.size()[0]):
        sample_points, sample_feats = sample_lines_feats(curves[j], lines_array, curve_feat[j], num)
        curve_samples.append(torch.unsqueeze(sample_points, 0))
        curve_sample_feats.append(torch.unsqueeze(sample_feats, 0))
    curve_samples = torch.cat(curve_samples, 0).contiguous()
    curve_sample_feats = torch.cat(curve_sample_feats, 0).contiguous()
    return curve_samples, curve_sample_feats

def batch_sample_triangles_feats(surfaces, faces_array, surface_feat, num):
    surface_samples = []
    surface_sample_feats = []
    for j in range(0, surfaces.size()[0]):
        sample_points, sample_feats = sample_triangles_feats(surfaces[j], faces_array, surface_feat[j], num)
        surface_samples.append(torch.unsqueeze(sample_points, 0))
        surface_sample_feats.append(torch.unsqueeze(sample_feats, 0))
    surface_samples = torch.cat(surface_samples, 0).contiguous()
    surface_sample_feats = torch.cat(surface_sample_feats, 0).contiguous()
    return surface_samples, surface_sample_feats

# how I sample from the lines
# verts = list of vertex positions : Nx3
# lines = list of lines, as vertex indecies in verts 
# feats = features of verts : CxN
# num = number of points to sample 
def sample_lines(verts, lines_array, num=2000): 
    lines = torch.tensor(lines_array, dtype=torch.int64).cuda()
    dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(verts.device), torch.tensor([1.0]).cuda(verts.device))

    # calculate area of each face 
    x1,x2,x3 = torch.split(torch.index_select(verts, 0, lines[:,0]) - torch.index_select(verts, 0, lines[:,1]), 1, dim = 1)
    Lengths = torch.sqrt(x1**2 + x2**2 + x3**2)
    Lengths = Lengths / torch.sum(Lengths) # percentage of each line w.r.t. full curve lines

    # define descrete distribution w.r.t. line area ratios caluclated 
    choices = np.arange(Lengths.shape[0])
    dist = stats.rv_discrete(name='custm', values=(choices, Lengths.data.cpu().numpy()))
    choices = dist.rvs(size=num) # list of lines to be sampled from 

    # from each face sample a point 
    select_lines = lines[choices]
    xs = torch.index_select(verts, 0, select_lines[:,0])
    ys = torch.index_select(verts, 0, select_lines[:,1])
    u = dist_uni.sample_n(num)
    points = (1- u)*xs + u*ys
    return points

def sample_triangles(verts, faces_array, num=10000): 
    faces = torch.tensor(faces_array, dtype=torch.int64).cuda()
    dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(verts.device), torch.tensor([1.0]).cuda(verts.device))

    # calculate area of each face 
    x1,x2,x3 = torch.split(torch.index_select(verts, 0,faces[:,0]) - torch.index_select(verts, 0,faces[:,1]), 1, dim = 1)
    y1,y2,y3 = torch.split(torch.index_select(verts, 0,faces[:,1]) - torch.index_select(verts, 0,faces[:,2]), 1, dim = 1)
    a = (x2*y3-x3*y2)**2
    b = (x3*y1 - x1*y3)**2
    c = (x1*y2 - x2*y1)**2
    Areas = torch.sqrt(a+b+c)/2
    Areas = Areas / torch.sum(Areas) # percentage of each face w.r.t. full surface area 

    # define descrete distribution w.r.t. face area ratios caluclated 
    choices = np.arange(Areas.shape[0])
    dist = stats.rv_discrete(name='custm', values=(choices, Areas.data.cpu().numpy()))
    choices = dist.rvs(size=num) # list of faces to be sampled from 

    # from each face sample a point 
    select_faces = faces[choices] 
    xs = torch.index_select(verts, 0, select_faces[:,0])
    ys = torch.index_select(verts, 0, select_faces[:,1])
    zs = torch.index_select(verts, 0, select_faces[:,2])
    u = torch.sqrt(dist_uni.sample_n(num))
    v = dist_uni.sample_n(num)
    points = (1- u)*xs + (u*(1-v))*ys + u*v*zs
    return points

def batch_sample_lines(curves, lines_array, num):
    curve_samples = []
    for j in range(0, curves.size()[0]):
        sample_points = sample_lines(curves[j], lines_array, num)
        curve_samples.append(torch.unsqueeze(sample_points, 0))
    curve_samples = torch.cat(curve_samples, 0).contiguous()
    return curve_samples

def batch_sample_triangles(surfaces, faces_array, num):
    surface_samples = []
    for j in range(0, surfaces.size()[0]):
        sample_points = sample_triangles(surfaces[j], faces_array, num)
        surface_samples.append(torch.unsqueeze(sample_points, 0))
    surface_samples = torch.cat(surface_samples, 0).contiguous()
    return surface_samples