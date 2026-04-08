# -*- coding: utf-8 -*-
class Args(dict):
    def __init__(self, n_cluster, file_name, network_type) -> None:
        self.encoded_space_dim = 50
        self.n_cluster = n_cluster
        self.num_epoch = 701
        self.learning_rate = 0.001
        self.LAMBDA = 1
        self.c = 20
        self.file_name = file_name
        self.network_type = network_type
        self.start_mf = 500
        self.cluster_reg_weight = 0.05
        self.diffuse_T = 8
        self.diff_rec_weight = 0.10
