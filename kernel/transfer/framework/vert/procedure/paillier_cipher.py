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

#

from common.python.utils import log_utils
from kernel.transfer.framework.vert.sync import paillier_keygen_sync

LOGGER = log_utils.get_logger()


class Provider(paillier_keygen_sync.Provider):

    def register_paillier_cipher(self, transfer_variables):
        self._register_paillier_keygen(pubkey_transfer=transfer_variables.paillier_pubkey)


class Promoter(paillier_keygen_sync.Promoter):

    def register_paillier_cipher(self, transfer_variables):
        self._register_paillier_keygen(pubkey_transfer=transfer_variables.paillier_pubkey)


class Arbiter(paillier_keygen_sync.Arbiter):

    def register_paillier_cipher(self, transfer_variables):
        self._register_paillier_keygen(pubkey_transfer=transfer_variables.paillier_pubkey)
