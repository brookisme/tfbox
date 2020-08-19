# tfbox

a collection of models and tools for tensorflow

--- 

TFBox's main utility lies within:

- [`tfbox.nn.encoder/decoder/encoder-decoder`](#models): an extremely flexible encoder/decoder python classes from which most modern architectures can be built using a simple (yaml) config file.

Additionally, TFBox contains a number of useful tools for TensorFlow, including:

- [an extremely flexible sequence class](#sequence) 
- [tools for model scoring](#scores) 
- [weighted metrics](#metrics) 
- [weighted loss function](#loss)
- [tensorboard callbacks](#tb)

---

##### INSTALL

`pip install tfbox`

##### ADDITIONAL REQUIREMENTS

- imagebox: https://github.com/brookisme/imagebox    
- TF>2
- numpy
- pandas
- pyyaml


<a name='#models'></a>

---

#### MODELS

`tfbox.nn.encoder/decoder/encoder-decoder` use yaml files to combine keras-model-blocks in `tfbox.nn.blocks` to build neural-networks.  The result is an flexible system from which you can build a large variety of models.  Lets start with some examples.

Here is the config for the [Xception Network](https://arxiv.org/abs/1610.02357):

xception:

```yaml
    blocks_config:
        - conv:
            filters: 32
            strides: 2
        - 64
        - stack:
            name: entry_flow_blocks
            seperable: true
            depth: 3
            output_stride: 2
            layers: [128,256,728]
        - stack:
            name: middle_flow
            nb_repeats: 16
            depth: 3
            filters: 728
        - stack:
            name: exit_flow_block
            output_stride: 2
            filters_list: [728,1024,1024]
        - stack:
            name: exit_flow_convs
            seperable: true
            residual: false
            layers: [1536,1536,2048]

        - aspp
```


<a name='#sequence'></a>

---

#### DFSequence

`tfbox.loaders.DFSequence` builds instances of `tf.keras.utils.Sequence` for image segmentation models using pandas dataframes. In particular it does almost anything you can imagine - but also can be bit overwhelming.

<a name='#scoring'></a>

---

#### SCORING



<a name='#metrics]'></a>


---

#### METRICS



<a name='#loss'></a>

---

#### LOSS FUNCTIONS



<a name='#callbacks'></a>

---

#### TENSORBOARD CALLBACKS



