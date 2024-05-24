import numpy as np
from scipy import stats

def sample_embeddings_labels(embeddings, labels, label_offset=0, samples_per_cluster=30):
    """sample embeddings for each label. label_offset is added to ensure uniqueness of label
    """
    sampled_embeddings = None
    labels += label_offset
    sampled_labels = None
    speaker_set = np.unique(labels)
    full_idx = np.indices([len(labels)])
    for i in speaker_set:
        speaker_idx = full_idx[0][labels==i]
        if len(speaker_idx) > samples_per_cluster:
            selected_speaker_idx = speaker_idx[:samples_per_cluster] #np.random.choice(speaker_idx, size=samples_per_cluster, replace=False)
        else:
            selected_speaker_idx = speaker_idx
        if sampled_embeddings is None:
            sampled_embeddings = embeddings[selected_speaker_idx].copy()
            sampled_labels = labels[selected_speaker_idx].copy()
        else:
            sampled_embeddings = np.concatenate((sampled_embeddings, embeddings[selected_speaker_idx].copy()), axis=0)
            sampled_labels = np.concatenate((sampled_labels, labels[selected_speaker_idx].copy()), axis=0)
    return sampled_embeddings, sampled_labels, labels

def reassign_labels(labels, old_sampled_labels, new_sampled_labels):
    old_sampled_labels = old_sampled_labels.copy()
    old_labels = labels.copy()
    new_sampled_labels = new_sampled_labels.copy()
    ## boosting the labels to ensure there is no overlap in label
    labels += np.max(new_sampled_labels) + 1
    for i in np.unique(old_sampled_labels):
        current_labels = new_sampled_labels[old_sampled_labels == i]
        m = stats.mode(current_labels, keepdims=True)[0][0]
        labels[old_labels==i] = m
    return labels
