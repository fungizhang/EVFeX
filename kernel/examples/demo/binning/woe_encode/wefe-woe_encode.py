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

import argparse
import os
import time

from kernel.examples.handler.component import DataIO, WoeEncode
from kernel.examples.handler.component import HorzBinning
from kernel.examples.handler.handler import Handler
from kernel.examples.handler.interface import Data
from kernel.examples.handler.utils.tools import load_job_config, JobConfig
from kernel.utils import consts


def main(config="../../config.yaml", param="./config.yaml", namespace="wefe_data"):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    promoter = parties.promoter[0]
    provider = parties.provider[0]
    arbiter = parties.arbiter[0]
    backend = config.backend
    work_mode = config.work_mode
    db_type = config.db_type
    data_base = config.data_base_dir

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    assert isinstance(param, dict)

    data_promoter = param["data_promoter"]
    data_provider = param["data_provider"]
    promoter_data_table = param.get("promoter_data_table")
    provider_data_table = param.get("provider_data_table")

    promoter_train_data = {"name": promoter_data_table, "namespace": namespace}
    provider_train_data = {"name": provider_data_table, "namespace": namespace}

    handler_upload = Handler().set_roles(promoter=promoter, provider=provider)
    handler_upload.add_upload_data(file=os.path.join(data_base, data_promoter),
                                   table_name=promoter_data_table,
                                   namespace=namespace,
                                   head=1, partition=1)
    handler_upload.add_upload_data(file=os.path.join(data_base, data_provider),
                                   table_name=provider_data_table,
                                   namespace=namespace,
                                   head=1, partition=1)
    handler_upload.upload(work_mode=work_mode, backend=backend, db_type=db_type)

    job_id = "job_" + time.strftime("%Y%m%d%H%M%S")
    # initialize handler
    handler = Handler(job_id=job_id, work_mode=work_mode, backend=backend, db_type=db_type, fl_type="horizontal")
    # set job initiator
    handler.set_initiator(role='promoter', member_id=promoter)
    # set participants information
    handler.set_roles(promoter=promoter, provider=provider, arbiter=arbiter)

    # define DataIO components
    dataio_0 = DataIO(name="dataio_0")  # start component numbering at 0
    # get DataIO member instance of promoter
    dataio_0_promoter_member_instance = dataio_0.get_member_instance(role='promoter', member_id=promoter)
    # configure DataIO for promoter
    dataio_0_promoter_member_instance.component_param(table=promoter_train_data, with_label=True, output_format="dense")
    # get and configure DataIO member instance of provider
    dataio_0.get_member_instance(role='provider', member_id=provider).component_param(table=provider_train_data,
                                                                                      with_label=True)

    binning_params = {
        'sample_bins': 100,
        'method': consts.RECURSIVE_QUERY
    }
    horz_binning = HorzBinning(name="horz_binning_0", **binning_params)

    woe_encode_params = {
        'method': 'cc'
    }
    woe_encode = WoeEncode(name="woe_encode_0", **woe_encode_params)

    # add components to handler, in order of task execution
    handler.add_component(dataio_0, output_data_type=["train"])
    handler.add_component(horz_binning, data=Data(train_data=dataio_0.name), output_data_type=["train"])
    handler.add_component(woe_encode, data=Data(train_data=dataio_0.name), output_data_type=['train'])

    # compile handler once finished adding modules, this step will form conf and dsl files for running job
    handler.compile()

    # fit model
    handler.fit()
    # query component summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WEFE woe encode JOB")
    parser.add_argument("-config", type=str,
                        help="config file", default="../../../config.yaml")
    parser.add_argument("-param", type=str,
                        help="config file for params", default="./config.yaml")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config, args.param)
    else:
        main()
