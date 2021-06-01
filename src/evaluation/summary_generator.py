import numpy as np
from knapsack_implementation import knapSack

def upsample_summary(summary_scores, n_frames, positions, change_points,knapsack):
    # compute the importance scores for the original length frame sequence
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    # fill the original length summary with the values of generated summary
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(summary_scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = summary_scores[i]

    if knapsack:
        print('with knapsack')
        summary,shot_lengths, selected_shot_idxs = generate_summary(frame_scores, change_points)
        return summary,shot_lengths, selected_shot_idxs
    else:
        return frame_scores, None, None


def generate_summary(frame_scores, change_points):
    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    shot_imp_scores = []
    shot_lengths = []
    for shot in change_points:
        shot_lengths.append(shot[1] - shot[0] + 1)
        shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

    # Select the best shots using the knapsack implementation
    final_max_length = int((shot[1] + 1) * 0.15)

    selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

    # Select all frames from each selected shot (by setting their value in the summary vector to 1)
    summary = np.zeros(shot[1] + 1, dtype=np.int8)
    for shot in selected:
        summary[change_points[shot][0]:change_points[shot][1] + 1] = 1

    return summary, shot_lengths,selected