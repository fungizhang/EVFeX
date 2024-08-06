# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: boosting-tree-model-meta.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='boosting-tree-model-meta.proto',
    package='com.welab.wefe.core.mlmodel.buffer',
    syntax='proto3',
    serialized_options=b'B\027BoostTreeModelMetaProto',
    serialized_pb=b'\n\x1e\x62oosting-tree-model-meta.proto\x12\"com.welab.wefe.core.mlmodel.buffer\"1\n\rObjectiveMeta\x12\x11\n\tobjective\x18\x01 \x01(\t\x12\r\n\x05param\x18\x02 \x03(\x01\"B\n\rCriterionMeta\x12\x18\n\x10\x63riterion_method\x18\x01 \x01(\t\x12\x17\n\x0f\x63riterion_param\x18\x02 \x03(\x01\"\xf0\x01\n\x15\x44\x65\x63isionTreeModelMeta\x12I\n\x0e\x63riterion_meta\x18\x01 \x01(\x0b\x32\x31.com.welab.wefe.core.mlmodel.buffer.CriterionMeta\x12\x11\n\tmax_depth\x18\x02 \x01(\x05\x12\x18\n\x10min_sample_split\x18\x03 \x01(\x05\x12\x1a\n\x12min_impurity_split\x18\x04 \x01(\x01\x12\x15\n\rmin_leaf_node\x18\x05 \x01(\x05\x12\x13\n\x0buse_missing\x18\x06 \x01(\x08\x12\x17\n\x0fzero_as_missing\x18\x07 \x01(\x08\"8\n\x0cQuantileMeta\x12\x17\n\x0fquantile_method\x18\x01 \x01(\t\x12\x0f\n\x07\x62in_num\x18\x02 \x01(\x05\"\x9e\x03\n\x15\x42oostingTreeModelMeta\x12L\n\ttree_meta\x18\x01 \x01(\x0b\x32\x39.com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta\x12\x15\n\rlearning_rate\x18\x02 \x01(\x01\x12\x11\n\tnum_trees\x18\x03 \x01(\x05\x12G\n\rquantile_meta\x18\x04 \x01(\x0b\x32\x30.com.welab.wefe.core.mlmodel.buffer.QuantileMeta\x12I\n\x0eobjective_meta\x18\x05 \x01(\x0b\x32\x31.com.welab.wefe.core.mlmodel.buffer.ObjectiveMeta\x12\x11\n\ttask_type\x18\x06 \x01(\t\x12\x18\n\x10n_iter_no_change\x18\x07 \x01(\x08\x12\x0b\n\x03tol\x18\x08 \x01(\x01\x12\x13\n\x0buse_missing\x18\t \x01(\x08\x12\x17\n\x0fzero_as_missing\x18\n \x01(\x08\x12\x11\n\twork_mode\x18\x0b \x01(\tB\x19\x42\x17\x42oostTreeModelMetaProtob\x06proto3'
)

_OBJECTIVEMETA = _descriptor.Descriptor(
    name='ObjectiveMeta',
    full_name='com.welab.wefe.core.mlmodel.buffer.ObjectiveMeta',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='objective', full_name='com.welab.wefe.core.mlmodel.buffer.ObjectiveMeta.objective', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='param', full_name='com.welab.wefe.core.mlmodel.buffer.ObjectiveMeta.param', index=1,
            number=2, type=1, cpp_type=5, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=70,
    serialized_end=119,
)

_CRITERIONMETA = _descriptor.Descriptor(
    name='CriterionMeta',
    full_name='com.welab.wefe.core.mlmodel.buffer.CriterionMeta',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='criterion_method', full_name='com.welab.wefe.core.mlmodel.buffer.CriterionMeta.criterion_method',
            index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='criterion_param', full_name='com.welab.wefe.core.mlmodel.buffer.CriterionMeta.criterion_param',
            index=1,
            number=2, type=1, cpp_type=5, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=121,
    serialized_end=187,
)

_DECISIONTREEMODELMETA = _descriptor.Descriptor(
    name='DecisionTreeModelMeta',
    full_name='com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='criterion_meta', full_name='com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta.criterion_meta',
            index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='max_depth', full_name='com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta.max_depth', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='min_sample_split',
            full_name='com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta.min_sample_split', index=2,
            number=3, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='min_impurity_split',
            full_name='com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta.min_impurity_split', index=3,
            number=4, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='min_leaf_node', full_name='com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta.min_leaf_node',
            index=4,
            number=5, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='use_missing', full_name='com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta.use_missing',
            index=5,
            number=6, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='zero_as_missing',
            full_name='com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta.zero_as_missing', index=6,
            number=7, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=190,
    serialized_end=430,
)

_QUANTILEMETA = _descriptor.Descriptor(
    name='QuantileMeta',
    full_name='com.welab.wefe.core.mlmodel.buffer.QuantileMeta',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='quantile_method', full_name='com.welab.wefe.core.mlmodel.buffer.QuantileMeta.quantile_method',
            index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='bin_num', full_name='com.welab.wefe.core.mlmodel.buffer.QuantileMeta.bin_num', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=432,
    serialized_end=488,
)

_BOOSTINGTREEMODELMETA = _descriptor.Descriptor(
    name='BoostingTreeModelMeta',
    full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='tree_meta', full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.tree_meta', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='learning_rate', full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.learning_rate',
            index=1,
            number=2, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='num_trees', full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.num_trees', index=2,
            number=3, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='quantile_meta', full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.quantile_meta',
            index=3,
            number=4, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='objective_meta', full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.objective_meta',
            index=4,
            number=5, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='task_type', full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.task_type', index=5,
            number=6, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='n_iter_no_change',
            full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.n_iter_no_change', index=6,
            number=7, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='tol', full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.tol', index=7,
            number=8, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='use_missing', full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.use_missing',
            index=8,
            number=9, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='zero_as_missing',
            full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.zero_as_missing', index=9,
            number=10, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='work_mode', full_name='com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta.work_mode', index=10,
            number=11, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=491,
    serialized_end=905,
)

_DECISIONTREEMODELMETA.fields_by_name['criterion_meta'].message_type = _CRITERIONMETA
_BOOSTINGTREEMODELMETA.fields_by_name['tree_meta'].message_type = _DECISIONTREEMODELMETA
_BOOSTINGTREEMODELMETA.fields_by_name['quantile_meta'].message_type = _QUANTILEMETA
_BOOSTINGTREEMODELMETA.fields_by_name['objective_meta'].message_type = _OBJECTIVEMETA
DESCRIPTOR.message_types_by_name['ObjectiveMeta'] = _OBJECTIVEMETA
DESCRIPTOR.message_types_by_name['CriterionMeta'] = _CRITERIONMETA
DESCRIPTOR.message_types_by_name['DecisionTreeModelMeta'] = _DECISIONTREEMODELMETA
DESCRIPTOR.message_types_by_name['QuantileMeta'] = _QUANTILEMETA
DESCRIPTOR.message_types_by_name['BoostingTreeModelMeta'] = _BOOSTINGTREEMODELMETA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ObjectiveMeta = _reflection.GeneratedProtocolMessageType('ObjectiveMeta', (_message.Message,), {
    'DESCRIPTOR': _OBJECTIVEMETA,
    '__module__': 'boosting_tree_model_meta_pb2'
    # @@protoc_insertion_point(class_scope:com.welab.wefe.core.mlmodel.buffer.ObjectiveMeta)
})
_sym_db.RegisterMessage(ObjectiveMeta)

CriterionMeta = _reflection.GeneratedProtocolMessageType('CriterionMeta', (_message.Message,), {
    'DESCRIPTOR': _CRITERIONMETA,
    '__module__': 'boosting_tree_model_meta_pb2'
    # @@protoc_insertion_point(class_scope:com.welab.wefe.core.mlmodel.buffer.CriterionMeta)
})
_sym_db.RegisterMessage(CriterionMeta)

DecisionTreeModelMeta = _reflection.GeneratedProtocolMessageType('DecisionTreeModelMeta', (_message.Message,), {
    'DESCRIPTOR': _DECISIONTREEMODELMETA,
    '__module__': 'boosting_tree_model_meta_pb2'
    # @@protoc_insertion_point(class_scope:com.welab.wefe.core.mlmodel.buffer.DecisionTreeModelMeta)
})
_sym_db.RegisterMessage(DecisionTreeModelMeta)

QuantileMeta = _reflection.GeneratedProtocolMessageType('QuantileMeta', (_message.Message,), {
    'DESCRIPTOR': _QUANTILEMETA,
    '__module__': 'boosting_tree_model_meta_pb2'
    # @@protoc_insertion_point(class_scope:com.welab.wefe.core.mlmodel.buffer.QuantileMeta)
})
_sym_db.RegisterMessage(QuantileMeta)

BoostingTreeModelMeta = _reflection.GeneratedProtocolMessageType('BoostingTreeModelMeta', (_message.Message,), {
    'DESCRIPTOR': _BOOSTINGTREEMODELMETA,
    '__module__': 'boosting_tree_model_meta_pb2'
    # @@protoc_insertion_point(class_scope:com.welab.wefe.core.mlmodel.buffer.BoostingTreeModelMeta)
})
_sym_db.RegisterMessage(BoostingTreeModelMeta)

DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)