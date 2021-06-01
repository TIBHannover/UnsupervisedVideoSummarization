import numpy as np
import torch
from scipy.spatial.distance import cdist


def compute_reward(seq, actions, _center_frames, segments, ignore_far_sim=True, temp_dist_thre=20,
                   use_gpu=False):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:, pick_idxs.detach().cpu().numpy()]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())
    # print('reward_rep {}'.format(reward_rep))

    # reward_uniformity
    all_frames = torch.arange(n).view(-1, 1)
    segment_idxs = find_frame_segment(segments, all_frames)
    #segment_idxs = list(dict.fromkeys(segment_idxs))
    segment_centers = find_segment_center(segment_idxs, _center_frames)

    u_mat = cdist(all_frames, all_frames, 'minkowski', p=2.)
    u_mat = u_mat[:, pick_idxs]
    u_mat = u_mat[segment_centers, :]

    u_mat = np.amin(u_mat, axis=1, keepdims=True)
    reward_uni = np.exp(-u_mat.mean())
    # print('reward_uni {}'.format(reward_uni))

    # weights for summe
    weights = [0.46, 0.43, 0.11]

    # weights for tvsum
    # weights = [0.45, 0.4, 0.15]
    reward = (weights[0] * reward_uni + weights[1] * reward_rep + weights[2] * reward_div) * (1 / sum(weights))

    # combine the three rewards
    # reward = (reward_uni + reward_div + reward_rep) * 0.333

    # combine the two rewards
    # reward = (reward_div + reward_rep) * 0.5

    return reward


def find_frame_segment(segments, selected_frames):
    segment_idxs = []
    for frame in selected_frames.detach().cpu().numpy():
        for segment_idx in segments:
            # npwhere returns tuple, where [0] has the found indices
            if len(np.where(segments[segment_idx] == frame)[0]) > 0:
                segment_idxs.append(segment_idx)
                break

    return np.asarray(segment_idxs)


def find_segment_center(segment_idxs, centers):
    return [centers[segment_idx] for segment_idx in segment_idxs]

