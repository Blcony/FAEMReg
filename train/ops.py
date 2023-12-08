import os
import sys
import tensorflow as tf
import subprocess
from tensorflow.python.framework import ops

OP_NAMES = ['correlation', 'downsample']

cwd = os.getcwd()
os.chdir('/userhome/EM_registration/RegisNet/Code/modelRaw7-15_1/train/ops')


def compile(op=None):
    to_compile = OP_NAMES

    tf_inc = " ".join(tf.sysconfig.get_compile_flags())
    tf_lib = " ".join(tf.sysconfig.get_link_flags())

    for n in to_compile:
        base = n + "_op"
        fn_cu_cc = base + ".cu.cc"
        fn_cu_o = base + ".cu.o"
        fn_cc = base + ".cc"
        fn_o = base + ".o"
        fn_so = base + ".so"

        nvcc_cmd = "nvcc -std=c++11 -c -o {} {} {} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I /usr/local --expt-relaxed-constexpr"
        nvcc_cmd = nvcc_cmd.format(" ".join([fn_cu_o, fn_cu_cc]),
                                   tf_inc, tf_lib)
        subprocess.check_output(nvcc_cmd, shell=True)

        gcc_cmd = "g++ -std=c++11 -shared -o {} {} -fPIC -L /usr/local/cuda/lib64 -lcudart {} -O2 -D -GOOGLE_CUDA=1"
        gcc_cmd = gcc_cmd.format(" ".join([fn_so, fn_cu_o, fn_cc]), tf_inc, tf_lib)
        subprocess.check_output(gcc_cmd, shell=True)


if __name__ == "__main__":
    compile()

module = sys.modules[__name__]
for n in OP_NAMES:
    lib_path = '/userhome/EM_registration/RegisNet/Code/modelRaw7-15_1/train/ops/{}_op.so'.format(n)
    try:
        op_lib = tf.load_op_library(lib_path)
    except:
        compile(n)
        op_lib = tf.load_op_library(lib_path)
    setattr(module, '_' + n + '_module', op_lib)

os.chdir(cwd)


def correlation(first, second, **kwargs):
    return _correlation_module.correlation(first, second, **kwargs)[0]


downsample = _downsample_module.downsample


@ops.RegisterGradient("Correlation")
def _CorrelationGrad(op, in_grad, in_grad1, in_grad2):
    grad0, grad1 = _correlation_module.correlation_grad(
        in_grad, op.inputs[0], op.inputs[1],
        op.outputs[1], op.outputs[2],
        kernel_size=op.get_attr('kernel_size'),
        max_displacement=op.get_attr('max_displacement'),
        pad=op.get_attr('pad'),
        stride_1=op.get_attr('stride_1'),
        stride_2=op.get_attr('stride_2')
    )
    return [grad0, grad1]


ops.NotDifferentiable("Downsample")
