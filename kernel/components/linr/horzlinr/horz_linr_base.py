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

from common.python.utils import log_utils
from kernel.components.linr.base_linr_model import BaseLinRModel         # tby
from kernel.components.linr.linr_model_weight import LinRModelWeights    # tby
from kernel.components.linr.param import HorzLinearParam                 # tby
from kernel.optimizer import activation
from kernel.optimizer.optimizer import optimizer_factory
from kernel.protobuf.generated import lr_model_meta_pb2
from kernel.transfer.variables.transfer_class.horz_linr_transfer_variable import HorzLinRTransferVariable
from kernel.utils import base_operator
from kernel.utils import consts
from kernel.utils import data_util
from kernel.utils.base_operator import vec_dot           #tby
from kernel.optimizer.loss import LeastSquaredErrorLoss  #tby

LOGGER = log_utils.get_logger()


class HorzLinRBaseModel(BaseLinRModel):
    def __init__(self):
        super(HorzLinRBaseModel, self).__init__()
        self.model_name = 'HorzLinearRegression'
        self.model_param_name = 'HorzLinearRegressionParam'
        self.model_meta_name = 'HorzLinearRegressionMeta'
        self.mode = consts.HORZ
        self.model_param = HorzLinearParam()
        self.aggregator = None
        self.is_serving_model = True

    def _init_model(self, params):
        super(HorzLinRBaseModel, self)._init_model(params)

        self.transfer_variable = HorzLinRTransferVariable()
        # This object is created in non-oot mode because no arbiter is used in OOT mode
        # and no arbiter blocks when creating the following object
        if self.component_properties.federated_learning_mode is None:
            self.aggregator.register_aggregator(self.transfer_variable)
        self.optimizer = optimizer_factory(params)
        self.aggregate_iters = params.aggregate_iters

    @property
    def use_loss(self):
        if self.model_param.early_stop == 'weight_diff':
            return False
        return True

    def classify(self, predict_wx, threshold):
        """
        convert a probability table into a predicted class table.
        """

        # predict_wx = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)

        def predict(x):
            prob = activation.sigmoid(x)
            pred_label = 1 if prob > threshold else 0
            return prob, pred_label

        predict_table = predict_wx.mapValues(predict)
        return predict_table

    def _init_model_variables(self, data_instances):
        model_shape = data_util.get_features_shape(data_instances)

        LOGGER.info("Initialized model shape is {}".format(model_shape))

        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj,
                                        data_instance=data_instances)
        model_weights = LinRModelWeights(w, fit_intercept=self.fit_intercept)
        return model_weights

    if __name__ == '__main__':
        if not False:
            print('hahah')

    def _compute_loss(self, data_instances):
        # f = functools.partial(self.gradient_operator.compute_loss,
        #                       coef=self.model_weights.coef_,
        #                       intercept=self.model_weights.intercept_)
        Y = data_instances.mapValues(lambda instance: instance.label)                                                                  #tby
        y_hat = data_instances.mapValues(lambda v: vec_dot(v.features, self.model_weights.coef_) + self.model_weights.intercept_)      #tby
        loss = LeastSquaredErrorLoss.compute_loss(Y, y_hat)                                                                            #tby
        
        #loss = data_instances.mapPartitions(f).reduce(base_operator.reduce_add)   # 是否需要这一步
        loss_norm = self.optimizer.loss_norm(self.model_weights)
        if loss_norm is not None:
            loss += loss_norm
        loss /= data_instances.count()
        self.callback_loss(self.n_iter_, loss)
        self.loss_history.append(loss)
        return loss

    def _get_meta(self):
        meta_protobuf_obj = lr_model_meta_pb2.LRModelMeta(penalty=self.model_param.penalty,
                                                          tol=self.model_param.tol,
                                                          alpha=self.alpha,
                                                          optimizer=self.model_param.optimizer,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.model_param.learning_rate,
                                                          max_iter=self.max_iter,
                                                          early_stop=self.model_param.early_stop,
                                                          fit_intercept=self.fit_intercept,
                                                          re_encrypt_batches=None,
                                                          need_one_vs_rest=self.need_one_vs_rest)
        return meta_protobuf_obj
