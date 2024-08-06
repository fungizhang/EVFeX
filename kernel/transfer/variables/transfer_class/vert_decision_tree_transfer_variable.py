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



################################################################################
#
# AUTO GENERATED TRANSFER VARIABLE CLASS. DO NOT MODIFY
#
################################################################################

from kernel.transfer.variables.base_transfer_variable import BaseTransferVariables


# noinspection PyAttributeOutsideInit
class VertDecisionTreeTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.cipher_compressor_para = self._create_variable(name='cipher_compressor_para')
        self.dispatch_node_provider = self._create_variable(name='dispatch_node_provider')
        self.dispatch_node_provider_result = self._create_variable(name='dispatch_node_provider_result')
        self.encrypted_grad_and_hess = self._create_variable(name='encrypted_grad_and_hess')
        self.encrypted_splitinfo_provider = self._create_variable(name='encrypted_splitinfo_provider')
        self.federated_best_splitinfo_provider = self._create_variable(name='federated_best_splitinfo_provider')
        self.final_splitinfo_provider = self._create_variable(name='final_splitinfo_provider')
        self.node_positions = self._create_variable(name='node_positions')
        self.predict_data = self._create_variable(name='predict_data')
        self.predict_data_by_provider = self._create_variable(name='predict_data_by_provider')
        self.predict_finish_tag = self._create_variable(name='predict_finish_tag')
        self.provider_cur_to_split_node_num = self._create_variable(name='provider_cur_to_split_node_num')
        self.provider_leafs = self._create_variable(name='provider_leafs')
        self.sync_flag = self._create_variable(name='sync_flag')
        self.tree = self._create_variable(name='tree')
        self.tree_node_queue = self._create_variable(name='tree_node_queue')

        # zfj
        self.q_array_of_promoter = self._create_variable(name='q_array_of_promoter')
        self.w_array_of_provider = self._create_variable(name='w_array_of_provider')
        self.w_array_of_provider_1 = self._create_variable(name='w_array_of_provider_1')
        self.response_to_provider = self._create_variable(name='response_to_provider')
        self.response_to_promoter = self._create_variable(name='response_to_promoter')
        self.max_trans_num_of_provider = self._create_variable(name='max_trans_num_of_provider')
        self.acc_data_num_of_provider = self._create_variable(name='acc_data_num_of_provider')
        self.acc_data_num_of_provider_1 = self._create_variable(name='acc_data_num_of_provider')
        self.pid_map_of_provider = self._create_variable(name='pid_map_of_provider')
        self.dsource_of_promoter = self._create_variable(name='dsource_of_promoter')

