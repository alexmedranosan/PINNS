??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
?
pinn/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_namepinn/dense_8/kernel
{
'pinn/dense_8/kernel/Read/ReadVariableOpReadVariableOppinn/dense_8/kernel*
_output_shapes

: *
dtype0
z
pinn/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namepinn/dense_8/bias
s
%pinn/dense_8/bias/Read/ReadVariableOpReadVariableOppinn/dense_8/bias*
_output_shapes
:*
dtype0
~
pinn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namepinn/dense/kernel
w
%pinn/dense/kernel/Read/ReadVariableOpReadVariableOppinn/dense/kernel*
_output_shapes

: *
dtype0
v
pinn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namepinn/dense/bias
o
#pinn/dense/bias/Read/ReadVariableOpReadVariableOppinn/dense/bias*
_output_shapes
: *
dtype0
?
pinn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_namepinn/dense_1/kernel
{
'pinn/dense_1/kernel/Read/ReadVariableOpReadVariableOppinn/dense_1/kernel*
_output_shapes

:  *
dtype0
z
pinn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namepinn/dense_1/bias
s
%pinn/dense_1/bias/Read/ReadVariableOpReadVariableOppinn/dense_1/bias*
_output_shapes
: *
dtype0
?
pinn/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_namepinn/dense_2/kernel
{
'pinn/dense_2/kernel/Read/ReadVariableOpReadVariableOppinn/dense_2/kernel*
_output_shapes

:  *
dtype0
z
pinn/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namepinn/dense_2/bias
s
%pinn/dense_2/bias/Read/ReadVariableOpReadVariableOppinn/dense_2/bias*
_output_shapes
: *
dtype0
?
pinn/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_namepinn/dense_3/kernel
{
'pinn/dense_3/kernel/Read/ReadVariableOpReadVariableOppinn/dense_3/kernel*
_output_shapes

:  *
dtype0
z
pinn/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namepinn/dense_3/bias
s
%pinn/dense_3/bias/Read/ReadVariableOpReadVariableOppinn/dense_3/bias*
_output_shapes
: *
dtype0
?
pinn/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_namepinn/dense_4/kernel
{
'pinn/dense_4/kernel/Read/ReadVariableOpReadVariableOppinn/dense_4/kernel*
_output_shapes

:  *
dtype0
z
pinn/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namepinn/dense_4/bias
s
%pinn/dense_4/bias/Read/ReadVariableOpReadVariableOppinn/dense_4/bias*
_output_shapes
: *
dtype0
?
pinn/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_namepinn/dense_5/kernel
{
'pinn/dense_5/kernel/Read/ReadVariableOpReadVariableOppinn/dense_5/kernel*
_output_shapes

:  *
dtype0
z
pinn/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namepinn/dense_5/bias
s
%pinn/dense_5/bias/Read/ReadVariableOpReadVariableOppinn/dense_5/bias*
_output_shapes
: *
dtype0
?
pinn/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_namepinn/dense_6/kernel
{
'pinn/dense_6/kernel/Read/ReadVariableOpReadVariableOppinn/dense_6/kernel*
_output_shapes

:  *
dtype0
z
pinn/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namepinn/dense_6/bias
s
%pinn/dense_6/bias/Read/ReadVariableOpReadVariableOppinn/dense_6/bias*
_output_shapes
: *
dtype0
?
pinn/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_namepinn/dense_7/kernel
{
'pinn/dense_7/kernel/Read/ReadVariableOpReadVariableOppinn/dense_7/kernel*
_output_shapes

:  *
dtype0
z
pinn/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namepinn/dense_7/bias
s
%pinn/dense_7/bias/Read/ReadVariableOpReadVariableOppinn/dense_7/bias*
_output_shapes
: *
dtype0

NoOpNoOp
?)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?(
value?(B?( B?(
?
dense_layers
dense_output
layer_sizes
sizes_w
sizes_b
U_0
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
8
0
1
2
3
4
5
6
7
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
 
 
 
 
?
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
(14
)15
16
17
?
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
(14
)15
16
17
 
?
*layer_regularization_losses
	variables
trainable_variables
+metrics
,non_trainable_variables
	regularization_losses
-layer_metrics

.layers
 
h

kernel
bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
h

kernel
bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
h

kernel
bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
h

 kernel
!bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
h

"kernel
#bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
h

$kernel
%bias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
h

&kernel
'bias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
h

(kernel
)bias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
WU
VARIABLE_VALUEpinn/dense_8/kernel.dense_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEpinn/dense_8/bias,dense_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Olayer_regularization_losses
trainable_variables
	variables
Pmetrics
Qnon_trainable_variables
regularization_losses
Rlayer_metrics

Slayers
MK
VARIABLE_VALUEpinn/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEpinn/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEpinn/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEpinn/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEpinn/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEpinn/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEpinn/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEpinn/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEpinn/dense_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEpinn/dense_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEpinn/dense_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEpinn/dense_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEpinn/dense_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEpinn/dense_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEpinn/dense_7/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEpinn/dense_7/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
?
0
1
2
3
4
5
6
7
8

0
1

0
1
 
?
Tlayer_regularization_losses
/trainable_variables
0	variables
Umetrics
Vnon_trainable_variables
1regularization_losses
Wlayer_metrics

Xlayers

0
1

0
1
 
?
Ylayer_regularization_losses
3trainable_variables
4	variables
Zmetrics
[non_trainable_variables
5regularization_losses
\layer_metrics

]layers

0
1

0
1
 
?
^layer_regularization_losses
7trainable_variables
8	variables
_metrics
`non_trainable_variables
9regularization_losses
alayer_metrics

blayers

 0
!1

 0
!1
 
?
clayer_regularization_losses
;trainable_variables
<	variables
dmetrics
enon_trainable_variables
=regularization_losses
flayer_metrics

glayers

"0
#1

"0
#1
 
?
hlayer_regularization_losses
?trainable_variables
@	variables
imetrics
jnon_trainable_variables
Aregularization_losses
klayer_metrics

llayers

$0
%1

$0
%1
 
?
mlayer_regularization_losses
Ctrainable_variables
D	variables
nmetrics
onon_trainable_variables
Eregularization_losses
player_metrics

qlayers

&0
'1

&0
'1
 
?
rlayer_regularization_losses
Gtrainable_variables
H	variables
smetrics
tnon_trainable_variables
Iregularization_losses
ulayer_metrics

vlayers

(0
)1

(0
)1
 
?
wlayer_regularization_losses
Ktrainable_variables
L	variables
xmetrics
ynon_trainable_variables
Mregularization_losses
zlayer_metrics

{layers
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1pinn/dense/kernelpinn/dense/biaspinn/dense_1/kernelpinn/dense_1/biaspinn/dense_2/kernelpinn/dense_2/biaspinn/dense_3/kernelpinn/dense_3/biaspinn/dense_4/kernelpinn/dense_4/biaspinn/dense_5/kernelpinn/dense_5/biaspinn/dense_6/kernelpinn/dense_6/biaspinn/dense_7/kernelpinn/dense_7/biaspinn/dense_8/kernelpinn/dense_8/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_55951511
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'pinn/dense_8/kernel/Read/ReadVariableOp%pinn/dense_8/bias/Read/ReadVariableOp%pinn/dense/kernel/Read/ReadVariableOp#pinn/dense/bias/Read/ReadVariableOp'pinn/dense_1/kernel/Read/ReadVariableOp%pinn/dense_1/bias/Read/ReadVariableOp'pinn/dense_2/kernel/Read/ReadVariableOp%pinn/dense_2/bias/Read/ReadVariableOp'pinn/dense_3/kernel/Read/ReadVariableOp%pinn/dense_3/bias/Read/ReadVariableOp'pinn/dense_4/kernel/Read/ReadVariableOp%pinn/dense_4/bias/Read/ReadVariableOp'pinn/dense_5/kernel/Read/ReadVariableOp%pinn/dense_5/bias/Read/ReadVariableOp'pinn/dense_6/kernel/Read/ReadVariableOp%pinn/dense_6/bias/Read/ReadVariableOp'pinn/dense_7/kernel/Read/ReadVariableOp%pinn/dense_7/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_55951768
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamepinn/dense_8/kernelpinn/dense_8/biaspinn/dense/kernelpinn/dense/biaspinn/dense_1/kernelpinn/dense_1/biaspinn/dense_2/kernelpinn/dense_2/biaspinn/dense_3/kernelpinn/dense_3/biaspinn/dense_4/kernelpinn/dense_4/biaspinn/dense_5/kernelpinn/dense_5/biaspinn/dense_6/kernelpinn/dense_6/biaspinn/dense_7/kernelpinn/dense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_55951832??
?	
?
E__inference_dense_5_layer_call_and_return_conditional_losses_55951642

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_55951220

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_5_layer_call_and_return_conditional_losses_55951328

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_8_layer_call_and_return_conditional_losses_55951522

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_55951582

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_55951662

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

*__inference_dense_7_layer_call_fn_55951691

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_559513822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
'__inference_pinn_layer_call_fn_55951468
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_pinn_layer_call_and_return_conditional_losses_559514262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_55951562

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?/
?
B__inference_pinn_layer_call_and_return_conditional_losses_55951426
input_1
dense_55951204
dense_55951206
dense_1_55951231
dense_1_55951233
dense_2_55951258
dense_2_55951260
dense_3_55951285
dense_3_55951287
dense_4_55951312
dense_4_55951314
dense_5_55951339
dense_5_55951341
dense_6_55951366
dense_6_55951368
dense_7_55951393
dense_7_55951395
dense_8_55951420
dense_8_55951422
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_55951204dense_55951206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_559511932
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_55951231dense_1_55951233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_559512202!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_55951258dense_2_55951260*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_559512472!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_55951285dense_3_55951287*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_559512742!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_55951312dense_4_55951314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_559513012!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_55951339dense_5_55951341*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_559513282!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_55951366dense_6_55951368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_559513552!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_55951393dense_7_55951395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_559513822!
dense_7/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_55951420dense_8_55951422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_559514092!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

*__inference_dense_5_layer_call_fn_55951651

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_559513282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

*__inference_dense_6_layer_call_fn_55951671

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_559513552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_8_layer_call_and_return_conditional_losses_55951409

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

*__inference_dense_4_layer_call_fn_55951631

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_559513012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_7_layer_call_and_return_conditional_losses_55951382

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_55951247

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_55951355

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_55951622

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
&__inference_signature_wrapper_55951511
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_559511782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_55951602

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

*__inference_dense_8_layer_call_fn_55951531

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_559514092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?K
?	
$__inference__traced_restore_55951832
file_prefix(
$assignvariableop_pinn_dense_8_kernel(
$assignvariableop_1_pinn_dense_8_bias(
$assignvariableop_2_pinn_dense_kernel&
"assignvariableop_3_pinn_dense_bias*
&assignvariableop_4_pinn_dense_1_kernel(
$assignvariableop_5_pinn_dense_1_bias*
&assignvariableop_6_pinn_dense_2_kernel(
$assignvariableop_7_pinn_dense_2_bias*
&assignvariableop_8_pinn_dense_3_kernel(
$assignvariableop_9_pinn_dense_3_bias+
'assignvariableop_10_pinn_dense_4_kernel)
%assignvariableop_11_pinn_dense_4_bias+
'assignvariableop_12_pinn_dense_5_kernel)
%assignvariableop_13_pinn_dense_5_bias+
'assignvariableop_14_pinn_dense_6_kernel)
%assignvariableop_15_pinn_dense_6_bias+
'assignvariableop_16_pinn_dense_7_kernel)
%assignvariableop_17_pinn_dense_7_bias
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B.dense_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB,dense_output/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_pinn_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_pinn_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_pinn_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_pinn_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_pinn_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_pinn_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_pinn_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_pinn_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_pinn_dense_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp$assignvariableop_9_pinn_dense_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_pinn_dense_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_pinn_dense_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp'assignvariableop_12_pinn_dense_5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp%assignvariableop_13_pinn_dense_5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_pinn_dense_6_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp%assignvariableop_15_pinn_dense_6_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_pinn_dense_7_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_pinn_dense_7_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18?
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*]
_input_shapesL
J: ::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

*__inference_dense_1_layer_call_fn_55951571

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_559512202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_7_layer_call_and_return_conditional_losses_55951682

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_55951193

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_55951542

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?Z
?
#__inference__wrapped_model_55951178
input_1-
)pinn_dense_matmul_readvariableop_resource.
*pinn_dense_biasadd_readvariableop_resource/
+pinn_dense_1_matmul_readvariableop_resource0
,pinn_dense_1_biasadd_readvariableop_resource/
+pinn_dense_2_matmul_readvariableop_resource0
,pinn_dense_2_biasadd_readvariableop_resource/
+pinn_dense_3_matmul_readvariableop_resource0
,pinn_dense_3_biasadd_readvariableop_resource/
+pinn_dense_4_matmul_readvariableop_resource0
,pinn_dense_4_biasadd_readvariableop_resource/
+pinn_dense_5_matmul_readvariableop_resource0
,pinn_dense_5_biasadd_readvariableop_resource/
+pinn_dense_6_matmul_readvariableop_resource0
,pinn_dense_6_biasadd_readvariableop_resource/
+pinn_dense_7_matmul_readvariableop_resource0
,pinn_dense_7_biasadd_readvariableop_resource/
+pinn_dense_8_matmul_readvariableop_resource0
,pinn_dense_8_biasadd_readvariableop_resource
identity??!pinn/dense/BiasAdd/ReadVariableOp? pinn/dense/MatMul/ReadVariableOp?#pinn/dense_1/BiasAdd/ReadVariableOp?"pinn/dense_1/MatMul/ReadVariableOp?#pinn/dense_2/BiasAdd/ReadVariableOp?"pinn/dense_2/MatMul/ReadVariableOp?#pinn/dense_3/BiasAdd/ReadVariableOp?"pinn/dense_3/MatMul/ReadVariableOp?#pinn/dense_4/BiasAdd/ReadVariableOp?"pinn/dense_4/MatMul/ReadVariableOp?#pinn/dense_5/BiasAdd/ReadVariableOp?"pinn/dense_5/MatMul/ReadVariableOp?#pinn/dense_6/BiasAdd/ReadVariableOp?"pinn/dense_6/MatMul/ReadVariableOp?#pinn/dense_7/BiasAdd/ReadVariableOp?"pinn/dense_7/MatMul/ReadVariableOp?#pinn/dense_8/BiasAdd/ReadVariableOp?"pinn/dense_8/MatMul/ReadVariableOp?
 pinn/dense/MatMul/ReadVariableOpReadVariableOp)pinn_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 pinn/dense/MatMul/ReadVariableOp?
pinn/dense/MatMulMatMulinput_1(pinn/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense/MatMul?
!pinn/dense/BiasAdd/ReadVariableOpReadVariableOp*pinn_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!pinn/dense/BiasAdd/ReadVariableOp?
pinn/dense/BiasAddBiasAddpinn/dense/MatMul:product:0)pinn/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense/BiasAddy
pinn/dense/TanhTanhpinn/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
pinn/dense/Tanh?
"pinn/dense_1/MatMul/ReadVariableOpReadVariableOp+pinn_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"pinn/dense_1/MatMul/ReadVariableOp?
pinn/dense_1/MatMulMatMulpinn/dense/Tanh:y:0*pinn/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_1/MatMul?
#pinn/dense_1/BiasAdd/ReadVariableOpReadVariableOp,pinn_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#pinn/dense_1/BiasAdd/ReadVariableOp?
pinn/dense_1/BiasAddBiasAddpinn/dense_1/MatMul:product:0+pinn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_1/BiasAdd
pinn/dense_1/TanhTanhpinn/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_1/Tanh?
"pinn/dense_2/MatMul/ReadVariableOpReadVariableOp+pinn_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"pinn/dense_2/MatMul/ReadVariableOp?
pinn/dense_2/MatMulMatMulpinn/dense_1/Tanh:y:0*pinn/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_2/MatMul?
#pinn/dense_2/BiasAdd/ReadVariableOpReadVariableOp,pinn_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#pinn/dense_2/BiasAdd/ReadVariableOp?
pinn/dense_2/BiasAddBiasAddpinn/dense_2/MatMul:product:0+pinn/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_2/BiasAdd
pinn/dense_2/TanhTanhpinn/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_2/Tanh?
"pinn/dense_3/MatMul/ReadVariableOpReadVariableOp+pinn_dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"pinn/dense_3/MatMul/ReadVariableOp?
pinn/dense_3/MatMulMatMulpinn/dense_2/Tanh:y:0*pinn/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_3/MatMul?
#pinn/dense_3/BiasAdd/ReadVariableOpReadVariableOp,pinn_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#pinn/dense_3/BiasAdd/ReadVariableOp?
pinn/dense_3/BiasAddBiasAddpinn/dense_3/MatMul:product:0+pinn/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_3/BiasAdd
pinn/dense_3/TanhTanhpinn/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_3/Tanh?
"pinn/dense_4/MatMul/ReadVariableOpReadVariableOp+pinn_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"pinn/dense_4/MatMul/ReadVariableOp?
pinn/dense_4/MatMulMatMulpinn/dense_3/Tanh:y:0*pinn/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_4/MatMul?
#pinn/dense_4/BiasAdd/ReadVariableOpReadVariableOp,pinn_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#pinn/dense_4/BiasAdd/ReadVariableOp?
pinn/dense_4/BiasAddBiasAddpinn/dense_4/MatMul:product:0+pinn/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_4/BiasAdd
pinn/dense_4/TanhTanhpinn/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_4/Tanh?
"pinn/dense_5/MatMul/ReadVariableOpReadVariableOp+pinn_dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"pinn/dense_5/MatMul/ReadVariableOp?
pinn/dense_5/MatMulMatMulpinn/dense_4/Tanh:y:0*pinn/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_5/MatMul?
#pinn/dense_5/BiasAdd/ReadVariableOpReadVariableOp,pinn_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#pinn/dense_5/BiasAdd/ReadVariableOp?
pinn/dense_5/BiasAddBiasAddpinn/dense_5/MatMul:product:0+pinn/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_5/BiasAdd
pinn/dense_5/TanhTanhpinn/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_5/Tanh?
"pinn/dense_6/MatMul/ReadVariableOpReadVariableOp+pinn_dense_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"pinn/dense_6/MatMul/ReadVariableOp?
pinn/dense_6/MatMulMatMulpinn/dense_5/Tanh:y:0*pinn/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_6/MatMul?
#pinn/dense_6/BiasAdd/ReadVariableOpReadVariableOp,pinn_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#pinn/dense_6/BiasAdd/ReadVariableOp?
pinn/dense_6/BiasAddBiasAddpinn/dense_6/MatMul:product:0+pinn/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_6/BiasAdd
pinn/dense_6/TanhTanhpinn/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_6/Tanh?
"pinn/dense_7/MatMul/ReadVariableOpReadVariableOp+pinn_dense_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"pinn/dense_7/MatMul/ReadVariableOp?
pinn/dense_7/MatMulMatMulpinn/dense_6/Tanh:y:0*pinn/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_7/MatMul?
#pinn/dense_7/BiasAdd/ReadVariableOpReadVariableOp,pinn_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#pinn/dense_7/BiasAdd/ReadVariableOp?
pinn/dense_7/BiasAddBiasAddpinn/dense_7/MatMul:product:0+pinn/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_7/BiasAdd
pinn/dense_7/TanhTanhpinn/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
pinn/dense_7/Tanh?
"pinn/dense_8/MatMul/ReadVariableOpReadVariableOp+pinn_dense_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"pinn/dense_8/MatMul/ReadVariableOp?
pinn/dense_8/MatMulMatMulpinn/dense_7/Tanh:y:0*pinn/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
pinn/dense_8/MatMul?
#pinn/dense_8/BiasAdd/ReadVariableOpReadVariableOp,pinn_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#pinn/dense_8/BiasAdd/ReadVariableOp?
pinn/dense_8/BiasAddBiasAddpinn/dense_8/MatMul:product:0+pinn/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
pinn/dense_8/BiasAdd
pinn/dense_8/TanhTanhpinn/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
pinn/dense_8/Tanh?
IdentityIdentitypinn/dense_8/Tanh:y:0"^pinn/dense/BiasAdd/ReadVariableOp!^pinn/dense/MatMul/ReadVariableOp$^pinn/dense_1/BiasAdd/ReadVariableOp#^pinn/dense_1/MatMul/ReadVariableOp$^pinn/dense_2/BiasAdd/ReadVariableOp#^pinn/dense_2/MatMul/ReadVariableOp$^pinn/dense_3/BiasAdd/ReadVariableOp#^pinn/dense_3/MatMul/ReadVariableOp$^pinn/dense_4/BiasAdd/ReadVariableOp#^pinn/dense_4/MatMul/ReadVariableOp$^pinn/dense_5/BiasAdd/ReadVariableOp#^pinn/dense_5/MatMul/ReadVariableOp$^pinn/dense_6/BiasAdd/ReadVariableOp#^pinn/dense_6/MatMul/ReadVariableOp$^pinn/dense_7/BiasAdd/ReadVariableOp#^pinn/dense_7/MatMul/ReadVariableOp$^pinn/dense_8/BiasAdd/ReadVariableOp#^pinn/dense_8/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2F
!pinn/dense/BiasAdd/ReadVariableOp!pinn/dense/BiasAdd/ReadVariableOp2D
 pinn/dense/MatMul/ReadVariableOp pinn/dense/MatMul/ReadVariableOp2J
#pinn/dense_1/BiasAdd/ReadVariableOp#pinn/dense_1/BiasAdd/ReadVariableOp2H
"pinn/dense_1/MatMul/ReadVariableOp"pinn/dense_1/MatMul/ReadVariableOp2J
#pinn/dense_2/BiasAdd/ReadVariableOp#pinn/dense_2/BiasAdd/ReadVariableOp2H
"pinn/dense_2/MatMul/ReadVariableOp"pinn/dense_2/MatMul/ReadVariableOp2J
#pinn/dense_3/BiasAdd/ReadVariableOp#pinn/dense_3/BiasAdd/ReadVariableOp2H
"pinn/dense_3/MatMul/ReadVariableOp"pinn/dense_3/MatMul/ReadVariableOp2J
#pinn/dense_4/BiasAdd/ReadVariableOp#pinn/dense_4/BiasAdd/ReadVariableOp2H
"pinn/dense_4/MatMul/ReadVariableOp"pinn/dense_4/MatMul/ReadVariableOp2J
#pinn/dense_5/BiasAdd/ReadVariableOp#pinn/dense_5/BiasAdd/ReadVariableOp2H
"pinn/dense_5/MatMul/ReadVariableOp"pinn/dense_5/MatMul/ReadVariableOp2J
#pinn/dense_6/BiasAdd/ReadVariableOp#pinn/dense_6/BiasAdd/ReadVariableOp2H
"pinn/dense_6/MatMul/ReadVariableOp"pinn/dense_6/MatMul/ReadVariableOp2J
#pinn/dense_7/BiasAdd/ReadVariableOp#pinn/dense_7/BiasAdd/ReadVariableOp2H
"pinn/dense_7/MatMul/ReadVariableOp"pinn/dense_7/MatMul/ReadVariableOp2J
#pinn/dense_8/BiasAdd/ReadVariableOp#pinn/dense_8/BiasAdd/ReadVariableOp2H
"pinn/dense_8/MatMul/ReadVariableOp"pinn/dense_8/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_55951301

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
}
(__inference_dense_layer_call_fn_55951551

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_559511932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_dense_2_layer_call_fn_55951591

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_559512472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_55951274

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

*__inference_dense_3_layer_call_fn_55951611

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_559512742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?,
?
!__inference__traced_save_55951768
file_prefix2
.savev2_pinn_dense_8_kernel_read_readvariableop0
,savev2_pinn_dense_8_bias_read_readvariableop0
,savev2_pinn_dense_kernel_read_readvariableop.
*savev2_pinn_dense_bias_read_readvariableop2
.savev2_pinn_dense_1_kernel_read_readvariableop0
,savev2_pinn_dense_1_bias_read_readvariableop2
.savev2_pinn_dense_2_kernel_read_readvariableop0
,savev2_pinn_dense_2_bias_read_readvariableop2
.savev2_pinn_dense_3_kernel_read_readvariableop0
,savev2_pinn_dense_3_bias_read_readvariableop2
.savev2_pinn_dense_4_kernel_read_readvariableop0
,savev2_pinn_dense_4_bias_read_readvariableop2
.savev2_pinn_dense_5_kernel_read_readvariableop0
,savev2_pinn_dense_5_bias_read_readvariableop2
.savev2_pinn_dense_6_kernel_read_readvariableop0
,savev2_pinn_dense_6_bias_read_readvariableop2
.savev2_pinn_dense_7_kernel_read_readvariableop0
,savev2_pinn_dense_7_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B.dense_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB,dense_output/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_pinn_dense_8_kernel_read_readvariableop,savev2_pinn_dense_8_bias_read_readvariableop,savev2_pinn_dense_kernel_read_readvariableop*savev2_pinn_dense_bias_read_readvariableop.savev2_pinn_dense_1_kernel_read_readvariableop,savev2_pinn_dense_1_bias_read_readvariableop.savev2_pinn_dense_2_kernel_read_readvariableop,savev2_pinn_dense_2_bias_read_readvariableop.savev2_pinn_dense_3_kernel_read_readvariableop,savev2_pinn_dense_3_bias_read_readvariableop.savev2_pinn_dense_4_kernel_read_readvariableop,savev2_pinn_dense_4_bias_read_readvariableop.savev2_pinn_dense_5_kernel_read_readvariableop,savev2_pinn_dense_5_bias_read_readvariableop.savev2_pinn_dense_6_kernel_read_readvariableop,savev2_pinn_dense_6_bias_read_readvariableop.savev2_pinn_dense_7_kernel_read_readvariableop,savev2_pinn_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : :: : :  : :  : :  : :  : :  : :  : :  : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$	 

_output_shapes

:  : 


_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
dense_layers
dense_output
layer_sizes
sizes_w
sizes_b
U_0
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
|__call__
}_default_save_signature
*~&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "PINN", "name": "pinn", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "PINN"}}
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25000, 32]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
(14
)15
16
17"
trackable_list_wrapper
?
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
(14
)15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
*layer_regularization_losses
	variables
trainable_variables
+metrics
,non_trainable_variables
	regularization_losses
-layer_metrics

.layers
|__call__
}_default_save_signature
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

kernel
bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25000, 2]}}
?

kernel
bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25000, 32]}}
?

kernel
bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25000, 32]}}
?

 kernel
!bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25000, 32]}}
?

"kernel
#bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25000, 32]}}
?

$kernel
%bias
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25000, 32]}}
?

&kernel
'bias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25000, 32]}}
?

(kernel
)bias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25000, 32]}}
%:# 2pinn/dense_8/kernel
:2pinn/dense_8/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Olayer_regularization_losses
trainable_variables
	variables
Pmetrics
Qnon_trainable_variables
regularization_losses
Rlayer_metrics

Slayers
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:! 2pinn/dense/kernel
: 2pinn/dense/bias
%:#  2pinn/dense_1/kernel
: 2pinn/dense_1/bias
%:#  2pinn/dense_2/kernel
: 2pinn/dense_2/bias
%:#  2pinn/dense_3/kernel
: 2pinn/dense_3/bias
%:#  2pinn/dense_4/kernel
: 2pinn/dense_4/bias
%:#  2pinn/dense_5/kernel
: 2pinn/dense_5/bias
%:#  2pinn/dense_6/kernel
: 2pinn/dense_6/bias
%:#  2pinn/dense_7/kernel
: 2pinn/dense_7/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tlayer_regularization_losses
/trainable_variables
0	variables
Umetrics
Vnon_trainable_variables
1regularization_losses
Wlayer_metrics

Xlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ylayer_regularization_losses
3trainable_variables
4	variables
Zmetrics
[non_trainable_variables
5regularization_losses
\layer_metrics

]layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^layer_regularization_losses
7trainable_variables
8	variables
_metrics
`non_trainable_variables
9regularization_losses
alayer_metrics

blayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
clayer_regularization_losses
;trainable_variables
<	variables
dmetrics
enon_trainable_variables
=regularization_losses
flayer_metrics

glayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
hlayer_regularization_losses
?trainable_variables
@	variables
imetrics
jnon_trainable_variables
Aregularization_losses
klayer_metrics

llayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mlayer_regularization_losses
Ctrainable_variables
D	variables
nmetrics
onon_trainable_variables
Eregularization_losses
player_metrics

qlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rlayer_regularization_losses
Gtrainable_variables
H	variables
smetrics
tnon_trainable_variables
Iregularization_losses
ulayer_metrics

vlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
wlayer_regularization_losses
Ktrainable_variables
L	variables
xmetrics
ynon_trainable_variables
Mregularization_losses
zlayer_metrics

{layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
'__inference_pinn_layer_call_fn_55951468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
#__inference__wrapped_model_55951178?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
B__inference_pinn_layer_call_and_return_conditional_losses_55951426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
*__inference_dense_8_layer_call_fn_55951531?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_8_layer_call_and_return_conditional_losses_55951522?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_55951511input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_55951551?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_55951542?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_55951571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_55951562?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_2_layer_call_fn_55951591?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_2_layer_call_and_return_conditional_losses_55951582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_3_layer_call_fn_55951611?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_3_layer_call_and_return_conditional_losses_55951602?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_4_layer_call_fn_55951631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_4_layer_call_and_return_conditional_losses_55951622?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_5_layer_call_fn_55951651?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_5_layer_call_and_return_conditional_losses_55951642?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_6_layer_call_fn_55951671?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_6_layer_call_and_return_conditional_losses_55951662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_7_layer_call_fn_55951691?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_7_layer_call_and_return_conditional_losses_55951682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_55951178{ !"#$%&'()0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
E__inference_dense_1_layer_call_and_return_conditional_losses_55951562\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? }
*__inference_dense_1_layer_call_fn_55951571O/?,
%?"
 ?
inputs????????? 
? "?????????? ?
E__inference_dense_2_layer_call_and_return_conditional_losses_55951582\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? }
*__inference_dense_2_layer_call_fn_55951591O/?,
%?"
 ?
inputs????????? 
? "?????????? ?
E__inference_dense_3_layer_call_and_return_conditional_losses_55951602\ !/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? }
*__inference_dense_3_layer_call_fn_55951611O !/?,
%?"
 ?
inputs????????? 
? "?????????? ?
E__inference_dense_4_layer_call_and_return_conditional_losses_55951622\"#/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? }
*__inference_dense_4_layer_call_fn_55951631O"#/?,
%?"
 ?
inputs????????? 
? "?????????? ?
E__inference_dense_5_layer_call_and_return_conditional_losses_55951642\$%/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? }
*__inference_dense_5_layer_call_fn_55951651O$%/?,
%?"
 ?
inputs????????? 
? "?????????? ?
E__inference_dense_6_layer_call_and_return_conditional_losses_55951662\&'/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? }
*__inference_dense_6_layer_call_fn_55951671O&'/?,
%?"
 ?
inputs????????? 
? "?????????? ?
E__inference_dense_7_layer_call_and_return_conditional_losses_55951682\()/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? }
*__inference_dense_7_layer_call_fn_55951691O()/?,
%?"
 ?
inputs????????? 
? "?????????? ?
E__inference_dense_8_layer_call_and_return_conditional_losses_55951522\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? }
*__inference_dense_8_layer_call_fn_55951531O/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_layer_call_and_return_conditional_losses_55951542\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? {
(__inference_dense_layer_call_fn_55951551O/?,
%?"
 ?
inputs?????????
? "?????????? ?
B__inference_pinn_layer_call_and_return_conditional_losses_55951426m !"#$%&'()0?-
&?#
!?
input_1?????????
? "%?"
?
0?????????
? ?
'__inference_pinn_layer_call_fn_55951468` !"#$%&'()0?-
&?#
!?
input_1?????????
? "???????????
&__inference_signature_wrapper_55951511? !"#$%&'();?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1?????????