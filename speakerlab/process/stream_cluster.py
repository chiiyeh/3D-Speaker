# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from speakerlab.process.cluster import AHCluster, SpectralCluster, UmapHdbscan
from speakerlab.utils.cluster_utils import sample_embeddings_labels, reassign_labels


class SequentialCluster:
    """
    Perform Sequential clustering
    """
    def __init__(self, max_num_spks=10, oracle_num=None, min_cluster_size=4, merge_delay=15, score_thres=0.6, sec_thres=0.5, speaker_thres=0.6, no_merge_gap=30, emb_dim=192):
        self.max_num_spks = max_num_spks
        self.min_cluster_size = min_cluster_size
        self.merge_delay = merge_delay
        self.score_thres = score_thres
        self.k = oracle_num
        self.sec_thres = sec_thres
        self.speaker_thres = speaker_thres
        self.no_merge_gap = no_merge_gap
        self.emb_dim = emb_dim
        max_speaker_allocation = max(100, 2*max_num_spks)
        self.cluster_details = {
            "avaliable_speakers": list(range(max_speaker_allocation)),
            "spk_counts": np.zeros((max_speaker_allocation, 1)),
            "spk_last_index": np.zeros((max_speaker_allocation, 1)),
            "spk_embeddings": np.zeros((max_speaker_allocation, emb_dim)),
            "labels": [],
        }

    def __call__(self, X):
        idx_offset = len(self.cluster_details['labels'])
        for idx, embedding in enumerate(X):
            idx += idx_offset
            if not self.cluster_details['labels']:
                label =  self.cluster_details['avaliable_speakers'].pop(0)
                self.cluster_details['spk_embeddings'][label] = embedding
                self.cluster_details['spk_last_index'][label] = idx
                self.cluster_details['spk_counts'][label] += 1
            else:
                scores = cosine_similarity(embedding[np.newaxis,:], self.cluster_details['spk_embeddings'])
                speakers_merge = []
                while scores.max() > self.score_thres:
                    label = scores.argmax()
                    speakers_merge.append(label)
                    scores[0][label] = 0
                if speakers_merge:
                    if len(speakers_merge) > 1:
                        ## Merge into main model only if they are close enough
                        final_speakers_merge = [speakers_merge[0]]
                        for lab in speakers_merge[1:]:
                            if cosine_similarity(self.cluster_details['spk_embeddings'][lab][np.newaxis], self.cluster_details['spk_embeddings'][speakers_merge[0]][np.newaxis]) >= self.sec_thres:
                                final_speakers_merge.append(lab)
                        speakers_merge = final_speakers_merge
                    if len(speakers_merge) == 1:
                        label = speakers_merge[0]
                        self.cluster_details['spk_embeddings'][label] = (self.cluster_details['spk_counts'][label] * self.cluster_details['spk_embeddings'][label] + embedding)/(self.cluster_details['spk_counts'][label]+1)
                        self.cluster_details['spk_counts'][label] += 1
                        self.cluster_details['spk_last_index'][label] = idx
                    else:
                        # print(f"Merging {len(speakers_merge)} speakers")
                        # print(f"current index: {idx}")
                        ## Not merging if gap exceeds threshold
                        label, speakers_merge = self.keep_label(speakers_merge)
                        speakers_merge = [lab for lab in speakers_merge if idx-self.get_first_idx(lab) <= self.no_merge_gap]

                        if speakers_merge:
                            merged_embedding, tot_count = self.get_merged_embedding_and_counts(speakers_merge + [label], embedding)
                            self.cluster_details['spk_embeddings'][label] = merged_embedding
                            self.cluster_details['spk_counts'][label] = tot_count
                            self.cluster_details['spk_last_index'][label] = idx
                            for lab in speakers_merge:
                                self.remove_speaker(lab)
                                _ = self.replace_speaker(label, lab)
                else:
                    label = self.cluster_details['avaliable_speakers'].pop(0)
                    self.cluster_details['spk_embeddings'][label] = embedding
                    self.cluster_details['spk_counts'][label] += 1
                    self.cluster_details['spk_last_index'][label] = idx
            self.cluster_details['labels'].append(label)

            if idx > 20:
                self.filter_minor_cluster_inter(self.merge_delay)
                self.merge_speaker_embeddings()
        return 
    
    def merge_speaker_embeddings(self):
        skip_merge = []
        idx = len(self.cluster_details['labels']) - 1
        while True:
            affinity = cosine_similarity(self.cluster_details['spk_embeddings'], self.cluster_details['spk_embeddings'])
            affinity = np.triu(affinity, 1)
            for i in skip_merge:
                affinity[i] = 0
            merge_idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[merge_idx] < self.speaker_thres:
                break
            # print(f"Speaker embedding merge: {affinity[merge_idx]}")
            c1, c2 = merge_idx
            if idx - self.get_first_idx(c1) > self.no_merge_gap and idx - self.get_first_idx(c2) > self.no_merge_gap:
                skip_merge.append(merge_idx)
                continue
            speakers_merge = [c1, c2]
            tot_count = 0
            merged_embedding = np.zeros(self.emb_dim)
            ## Get new speaker embeddings
            for lab in speakers_merge:
                tot_count += self.cluster_details['spk_counts'][lab]
                merged_embedding += self.cluster_details['spk_counts'][lab] * self.cluster_details['spk_embeddings'][lab]
            label, speakers_merge = self.keep_label(speakers_merge)
            self.cluster_details['spk_embeddings'][label] = merged_embedding / tot_count
            self.cluster_details['spk_counts'][label] = tot_count
            self.cluster_details['spk_last_index'][label] = max(
                self.cluster_details['spk_last_index'][label], 
                self.cluster_details['spk_last_index'][speakers_merge[0]]
                )
            for lab in speakers_merge:
                self.remove_speaker(lab)
                first_idx = self.replace_speaker(label, lab)
                if idx-first_idx > 20:
                    print(f"Merged gap {idx - first_idx}")
        return

    def get_merged_embedding_and_counts(self, speakers_merge,  embedding):
        total_spk_counts = 0
        current_embedding = np.zeros(self.emb_dim)
        ## Get new speaker embeddings
        for lab in speakers_merge:
            total_spk_counts += self.cluster_details['spk_counts'][lab]
            current_embedding += self.cluster_details['spk_counts'][lab] * self.cluster_details['spk_embeddings'][lab]
        merged_embedding = (current_embedding + embedding)/(total_spk_counts + 1)
        return merged_embedding, total_spk_counts+1

    def keep_label(self, speakers_merge):
        ## Keeping the label with the earliest appearance
        for lab in self.cluster_details['labels']:
            if lab in speakers_merge:
                break
        speakers_merge.remove(lab)
        return lab, speakers_merge
    
    def get_first_idx(self, speaker):
        for lab_idx, lab in enumerate(self.cluster_details['labels']):
            if lab == speaker:
                return lab_idx
        return -1
    
    def replace_speaker(self, speaker, old_speaker):
        first_idx = -1
        for lab_idx, lab in enumerate(self.cluster_details['labels']):
            if lab == old_speaker:
                if first_idx < 0:
                    first_idx = lab_idx
                self.cluster_details['labels'][lab_idx] = speaker
        return first_idx
    
    def remove_speaker(self, old_speaker):
        self.cluster_details['spk_embeddings'][old_speaker] = np.zeros(self.emb_dim)
        self.cluster_details['spk_last_index'][old_speaker] = 0
        self.cluster_details['spk_counts'][old_speaker] = 0
        self.cluster_details['avaliable_speakers'].insert(0, old_speaker)
        return 
    
    def filter_minor_cluster_inter(self, merge_delay):
        cset = set(self.cluster_details['labels'])
        minor_cset = []
        major_cset = []
        for lab in cset:
            if self.cluster_details['spk_counts'][lab] < self.min_cluster_size and self.cluster_details['spk_last_index'][lab] < len(self.cluster_details['labels']) - merge_delay:
                first_idx = self.get_first_idx(lab)
                if  len(self.cluster_details['labels']) - first_idx > self.no_merge_gap:
                    print(f"merging beyond {self.no_merge_gap}, {len(self.cluster_details['labels']) - first_idx}")
                minor_cset.append(lab)
            elif self.cluster_details['spk_counts'][lab] >= self.min_cluster_size:
                major_cset.append(lab)

        if len(minor_cset) == 0:
            return
        
        if len(major_cset) == 0:
            ## TODO: handle this case
            print("Gg")
            return

        for i in minor_cset:
            cos_sim = cosine_similarity(self.cluster_details['spk_embeddings'][i][np.newaxis], self.cluster_details['spk_embeddings'])
            cos_sim[0][i] = 0
            speaker = cos_sim.argmax()
            self.replace_speaker(speaker, i)
            self.remove_speaker(i)
        
        return

class CommonStreamClustering:
    """Perfom clustering for input embeddings and output the labels.
    """

    def __init__(self, cluster_type, cluster_line=10, mer_cos=None, min_cluster_size=4, buffer_size=30, samples_per_cluster=30, **kwargs):
        self.cluster_type = cluster_type
        self.cluster_line = cluster_line
        self.min_cluster_size = min_cluster_size
        self.mer_cos = mer_cos
        self.buffer_size = buffer_size
        self.samples_per_cluster = samples_per_cluster
        if self.cluster_type == 'spectral':
            self.cluster = SpectralCluster(**kwargs)
        elif self.cluster_type == 'umap_hdbscan':
            kwargs['min_cluster_size'] = min_cluster_size
            self.cluster = UmapHdbscan(**kwargs)
        elif self.cluster_type == 'AHC':
            self.cluster = AHCluster(**kwargs)
        else:
            raise ValueError(
                '%s is not currently supported.' % self.cluster_type
            )

        self.cluster_details = {
            "carried_embeddings": None,
            "sampled_embeddings": None,
            "sample_labels": None,
            "buffer_embeddings": None,
            "labels": [],
            "embeddings": None,
        }
    

    def __call__(self, X):
        ## Currently missing merging of cluster and filtering small cluster
        combined_ls = []
        if self.cluster_details["carried_embeddings"] is not None:
            combined_ls.append(self.cluster_details['carried_embeddings'])                   
        if self.cluster_details['sampled_embeddings'] is not None:
            combined_ls.append(self.cluster_details['sampled_embeddings'])
        if self.cluster_details['buffer_embeddings'] is not None:
            combined_ls.append(self.cluster_details['buffer_embeddings'])
        combined_ls.append(X)
        combined_embeddings = np.concatenate(combined_ls, axis=0)
        if len(combined_embeddings) < 10:
            self.cluster_details["carried_embeddings"] = combined_embeddings
            return
        else:
            self.cluster_details["carried_embeddings"] = None
        new_labels = self.cluster(combined_embeddings)

        ## Changing the label to be consistent with the sampled embeddings
        if self.cluster_details['sampled_embeddings'] is not None:
            new_labels = reassign_labels(new_labels, new_labels[:len(self.cluster_details['sampled_labels'])], self.cluster_details['sampled_labels'])
            self.cluster_details['labels'] = np.concatenate([self.cluster_details['labels'][:-self.buffer_size], new_labels[-self.buffer_size - len(X):]])
        else:
            self.cluster_details['labels'] = new_labels

        if len(new_labels) > self.buffer_size:
            ## should i use new_labels or current_labels
            self.cluster_details['sampled_embeddings'], self.cluster_details['sampled_labels'], _ = sample_embeddings_labels(combined_embeddings[:-self.buffer_size], new_labels[:-self.buffer_size], label_offset=0, samples_per_cluster=self.samples_per_cluster)
            self.cluster_details['buffer_embeddings'] = combined_embeddings[-self.buffer_size:]
        else:
            self.cluster_details['sampled_embeddings'] = None
            self.cluster_details['sampled_labels'] = None
            self.cluster_details['buffer_embeddings'] = combined_embeddings
        return

    def filter_minor_cluster_inter(self, merge_delay):
        """NOT IMPLEMENTED"""
        return