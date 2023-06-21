training of images

SUB:[Create a new env in conda] 
    conda create -n maddi python=3.7.4 ipython
SUB:[AttributeError: module 'google.protobuf.internal.containers' has no attribute 'MutableMapping]
     pip install protobuf==3.20.1
SUB:[AttributeError: type object 'h5py.h5.H5PYConfig' has no attribute '__reduce_cython__']
    pip install --upgrade h5py
SUB:[ImportError: cannot import name 'LayerNormalization' from 'tensorflow.python.keras.layers.normalization']
    pip install --upgrade tensorflow
SUB:[ModuleNotFoundError: No module named 'tensorflow.compat']
    pip install --upgrade tensorflow-gpu==2.2.0 --user
    pip uninstall tensorflow-datasets
    pip install tensorflow-datasets==4.0.0
SUB:[AttributeError: module 'tensorflow.compat.v2' has no attribute '__internal__']
    //import tensorflow.keras instead of keras


SUB:[path where maddi env is present]
    C:\users\manoj\.conda\envs\maddi\lib\site-packages\tensorflow>



Training of overlap

SUB:[cannot import name 'multiheadattention' from 'tensorflow.keras.layers' site:stackoverflow.com]
    //created a file named tsMultiHeadAttention and copied the code got github
    https://github.com/keras-team/keras/blob/v2.12.0/keras/layers/attention/multi_head_attention.py#L130-L730
    //should have to use tf.keras.layers.MultiHeadAttention. Not tfa.layers.MultiHeadAttention


SUB:[Create a requirements file in proper format]
    pip list --format=freeze > requirements.txt

SUB:[`tensorflow.python.framework.errors_impl.NotFoundError`]
    file path is incorrect,as there was some space in folder name



#############
No of patints
#############

mri 331




x_train
10234 rows × 19757 columns

x_test
4387 rows × 19757 columns

