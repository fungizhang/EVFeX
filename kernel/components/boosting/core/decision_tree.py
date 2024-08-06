#!/usr/bin/env python    
# -*- coding: utf-8 -*- 

# Copyright 2021 Tianmian Tech. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2019 The FATE Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import List

import numpy as np
import copy
import time # zfj
# import torch  #zfj: 使用GPU
# device = torch.device("cuda:1")

from common.python.utils import log_utils
from kernel.components.boosting.core.feature_histogram import \
    FeatureHistogram
from kernel.components.boosting.core.feature_importance import FeatureImportance
from kernel.components.boosting.core.node import Node
from kernel.components.boosting.core.splitter import Splitter
from kernel.utils import consts

LOGGER = log_utils.get_logger()


class DecisionTree(object):
    def __init__(self, tree_param):
        self.criterion_method = tree_param.criterion_method
        self.criterion_params = tree_param.criterion_params
        self.max_depth = tree_param.max_depth
        self.min_sample_split = tree_param.min_sample_split
        self.min_impurity_split = tree_param.min_impurity_split
        self.min_leaf_node = tree_param.min_leaf_node
        self.max_split_nodes = tree_param.max_split_nodes
        self.feature_importance_type = tree_param.feature_importance_type
        self.n_iter_no_change = tree_param.n_iter_no_change
        self.tol = tree_param.tol
        self.use_missing = tree_param.use_missing
        self.zero_as_missing = tree_param.zero_as_missing
        self.min_child_weight = tree_param.min_child_weight
        self.sitename = ''
        self.feature_importance = {}

        self.runtime_idx = None
        self.sitename = None
        self.valid_features = None
        self.transfer_inst = None
        self.data_bin = None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node, self.min_child_weight)
        self.sample_weights = None
        self.tree_ = []

        # histogram
        self.deterministic = tree_param.deterministic
        self.hist_computer = FeatureHistogram()
        if self.deterministic:
            self.hist_computer.stable_reduce = True

    def init_variables(self, flowid, runtime_idx, data_bin, bin_split_points, bin_sparse_points, valid_features):

        LOGGER.info("set flowid, flowid is {}".format(flowid))
        self.transfer_inst.set_flowid(flowid)
        self.runtime_idx = runtime_idx
        self.sitename = ":".join([self.sitename, str(self.runtime_idx)])

        LOGGER.info("set valid features")
        self.valid_features = valid_features
        self.data_bin = data_bin
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points

    def set_flowid(self, flowid=0):
        LOGGER.info("set flowid, flowid is {}".format(flowid))
        self.transfer_inst.set_flowid(flowid)

    def set_runtime_idx(self, runtime_idx):
        self.runtime_idx = runtime_idx
        self.sitename = ":".join([self.sitename, str(self.runtime_idx)])

    def set_valid_features(self, valid_features=None):
        LOGGER.info("set valid features")
        self.valid_features = valid_features

    def set_grad_and_hess(self, grad_and_hess):
        self.grad_and_hess = grad_and_hess

    def set_input_data(self, data_bin, bin_split_points, bin_sparse_points):
        self.data_bin = data_bin
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points

    def check_max_split_nodes(self):
        # check max_split_nodes
        if self.max_split_nodes != 0 and self.max_split_nodes % 2 == 1:
            self.max_split_nodes += 1
            LOGGER.warning('an even max_split_nodes value is suggested '
                           'when using histogram-subtraction, max_split_nodes reset to {}'.format(self.max_split_nodes))

    def fit(self):
        raise NotImplementedError("fit method should overload")

    def predict(self, data_inst):
        raise NotImplementedError("fit method should overload")

    def get_feature_importance(self):
        return self.feature_importance

    @staticmethod
    def get_node_map(nodes: List[Node], left_node_only=False):
        node_map = {}
        idx = 0
        for node in nodes:
            if node.id != 0 and (not node.is_left_node and left_node_only):
                continue
            node_map[node.id] = idx
            idx += 1
        return node_map

    @staticmethod
    def get_grad_hess_sum(grad_and_hess_table):
        LOGGER.info("calculate the sum of grad and hess")
        grad, hess = grad_and_hess_table.reduce(
            lambda value1, value2: (value1[0] + value2[0], value1[1] + value2[1]))
        return grad, hess

    @staticmethod
    def dispatch_all_node_to_root(data_bin, root_node_id):
        return data_bin.mapValues(lambda inst: (1, root_node_id))

    def update_feature_importance(self, splitinfo, record_site_name=True):

        inc_split, inc_gain = 1, splitinfo.gain

        sitename = splitinfo.sitename
        fid = splitinfo.best_fid

        if record_site_name:
            key = (sitename, fid)
        else:
            key = fid

        if key not in self.feature_importance:
            self.feature_importance[key] = FeatureImportance(0, 0, self.feature_importance_type)

        self.feature_importance[key].add_split(inc_split)
        if inc_gain is not None:
            self.feature_importance[key].add_gain(inc_gain)



    ### zfj:多次交互
    def get_local_histograms_zfj(self, trans_num, Z, sub_R, bin_nums, feature_nums, histogram_nums, data_bin_num_list):



        ZT = Z.T
        ZZT = Z.dot(ZT)
        # LOGGER.info("==============ZZT=========={}".format(ZZT))
        I = np.eye(Z.shape[0])
        IsubZZT = (I - ZZT)

        # LOGGER.info("==============IsubZZT=========={}".format(IsubZZT))

        ##### 构建PP端矩阵


        PP_array_feas_list = []
        for h in range(histogram_nums):
            PP_array_feas = []
            for i in range(feature_nums):
                PP_array = np.zeros((Z.shape[0], bin_nums[i]))
                for j in range(bin_nums[i]):
                    for idx in data_bin_num_list[h][i][j]:
                        PP_array[idx][j] = 1
                    # LOGGER.info("===========PP_array[:][j]============={}".format(PP_array[:,j]))
                    if j < bin_nums[i] - 1:
                        PP_array[:,j+1] = copy.deepcopy(PP_array[:,j])
                PP_array_feas.append(PP_array)
            PP_array_feas_list.append(PP_array_feas)

        W = []
        for h in range(histogram_nums):
            W_h = []
            for pp_array in PP_array_feas_list[h]:
                w_fea = IsubZZT.dot(pp_array)
                W_h.append(w_fea)
            W.append(W_h)

        # W = [np.array(W)]
        # LOGGER.info("==============W====111======{}".format(W))
        # LOGGER.info("==============W======111===={}".format(np.array(W).shape))

        # ZT = Z.T
        # sample_nums_fix = Z.shape[0]
        # if (trans_num+1) * sub_R > sample_nums_fix:
        #     Z_sub = Z[trans_num * sub_R : sample_nums_fix]
        #     row_num = sample_nums_fix - trans_num * sub_R
        # else:
        #     Z_sub = Z[trans_num * sub_R : (trans_num+1) * sub_R]
        #     row_num = sub_R
        #
        #
        # ###### CPU
        # Z_subZT = Z_sub.dot(ZT)
        # tmp = - Z_subZT
        #
        #
        #
        #
        # for i in range(row_num):
        #     tmp[i, trans_num * sub_R + i] += 1
        #
        # col_pre_rows = Z_sub.shape[0]
        #
        # W = [[] for _ in range(histogram_nums)]
        # for i in range(histogram_nums):
        #     for j in range(feature_nums):
        #         W[i].append(np.zeros((col_pre_rows, bin_nums[j])))
        #
        # start_time = time.time()
        # for h in range(histogram_nums):
        #     for f in range(feature_nums):
        #         col_pre = np.zeros(col_pre_rows)
        #         for b in range(bin_nums[f]):
        #             selected_columns = data_bin_num_list[h][f][b]
        #             if len(selected_columns) != 0:
        #                 col_pre += np.sum(tmp[:, selected_columns], axis=1)
        #             W[h][f][:, b] = col_pre
        #
        # end_time = time.time()
        # execution_time = end_time - start_time
        # LOGGER.info("==========计算W=============={}".format(execution_time))
        #
        # LOGGER.info("==============W====222======{}".format(W))
        # LOGGER.info("==============W======222===={}".format(np.array(W).shape))

        acc_histograms_pre = [W, trans_num, sub_R]
        return acc_histograms_pre

    ### zfj：多次交互
    def get_acc_data_num_zfj(self, bin_nums, feature_nums, histogram_nums, data_bin_num_list):

        acc_data_num = [[] for _ in range(histogram_nums)]
        for i in range(histogram_nums):
            for j in range(feature_nums):
                acc_data_num[i].append(np.zeros(bin_nums[j]))


        for h in range(histogram_nums):
            for f in range(feature_nums):
                data_num = 0
                for b in range(bin_nums[f]):
                    selected_columns = data_bin_num_list[h][f][b]
                    data_num += len(selected_columns)
                    acc_data_num[h][f][b] = data_num

        return [acc_data_num]

    ### zfj:多次交互
    def get_pid_zfj(self, cur_to_split_nodes):
        pid_map = self.hist_computer.compute_pid_zfj(cur_to_split_nodes)
        return pid_map


    def get_local_histograms(self, dep, data_with_pos, g_h, node_sample_count, cur_to_split_nodes, node_map,
                             ret='tensor', sparse_opt=False
                             , hist_sub=True, bin_num=None):

        LOGGER.info("start to compute node histograms")
        acc_histograms = self.hist_computer.compute_histogram(dep,
                                                              data_with_pos,
                                                              g_h,
                                                              self.bin_split_points,
                                                              self.bin_sparse_points,
                                                              self.valid_features,
                                                              node_map, node_sample_count,
                                                              use_missing=self.use_missing,
                                                              zero_as_missing=self.zero_as_missing,
                                                              ret=ret,
                                                              hist_sub=hist_sub,
                                                              sparse_optimization=sparse_opt,
                                                              cur_to_split_nodes=cur_to_split_nodes,
                                                              bin_num=bin_num)
        return acc_histograms

    @staticmethod
    def sample_count_map_func(kv, node_map):

        # record node sample number in count_arr
        count_arr = np.zeros(len(node_map))
        for k, v in kv:
            if v[1] not in node_map:
                continue
            node_idx = node_map[v[1]]  # node position
            count_arr[node_idx] += 1
        return count_arr

    @staticmethod
    def sample_count_reduce_func(v1, v2):
        return v1 + v2

    def count_node_sample_num(self, node_dispatch, node_map):
        """
        count sample number in every leaf node
        """
        count_func = functools.partial(self.sample_count_map_func, node_map=node_map)
        rs = node_dispatch.applyPartitions(count_func).reduce(self.sample_count_reduce_func)
        return rs

    def get_sample_weights(self):
        return self.sample_weights

    @staticmethod
    def assign_instance_to_root_node(data_bin, root_node_id):
        return data_bin.mapValues(lambda inst: (1, root_node_id))

    @staticmethod
    def float_round(num):
        """
        prevent float error
        """
        return round(num, consts.TREE_DECIMAL_ROUND)

    def round_leaf_val(self):
        # process predict weight to prevent float error
        for node in self.tree_:
            if node.is_leaf:
                node.weight = self.float_round(node.weight)
