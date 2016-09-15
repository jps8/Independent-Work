%%-- 03/01/2016 08:44:56 PM --%%
run examples/mnist/prepare_mnist.m
exit
%%-- 04/04/2016 07:41:09 PM --%%
run examples/mnist/prepare_mnist.m
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
q
quit()
%%-- 04/04/2016 07:42:05 PM --%%
run examples/mnist/prepare_mnist.m
quit()
%%-- 04/04/2016 07:44:29 PM --%%
run examples/mnist/prepare_data.m
run examples/mnist/prepare_mnist.m
quit()
