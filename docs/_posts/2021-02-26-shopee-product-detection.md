---
layout: post
title: Efficientnet-B5 on Shopee Product Dataset
author: Tu T. Do
toc: true
---

<!-- Mathjax Support -->
<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

This article describes training processes and experimentations done on a Kaggle dataset [Shopee Product Detection](#) as part of the class's assignements. All trainings and experiments were done on Google Colab.

## Dataset
The dataset consists of 42 classes, encoded from 0 to 41, partitioned into 2 subsets:
- Train: 100k images
- Test: 20k images


```
dataset
|__ resized/
|  |__ train/
|  |  |__ 01/
|  |  |  |__ img01.png
|  |  |  |__ ...
|  |  |  |__ img_xxx.png
|  |  |__ .../
|  |  |__ 41
|  |__ test/
|__ train.csv
|__ test.csv
```

## Enviroments:
- Google colab
- Python3.6
- Tensorflow 2.4.1


## Input pipeline
Input pipeline consists of following functions:
- Read images file

```python
PTH = <PTH_TO_DATASET>

def get_dataset(valid_size = .1):
    pad = lambda x: f"0{x}" if x < 10 else f"{x}"
    
    # Read train.csv
    train = pd.read_csv(f"{PTH}/train.csv")
    train_pth = train.apply(lambda x: \
        f"{PTH}/resized/train/{pad(x['category'])}/{x['filename']}", axis=1)
    train_lab = train["category"]
    
    # Read test.csv
    test = pd.read_csv(f"{PTH}/test.csv")
    test_pth = test.apply(lambda x: \
        f"{PTH}/resized/test/{x['filename']}", axis=1)
    test_lab = test["category"].values

    # Validation split:
    train_pth , valid_pth, train_lab, valid_lab = \
        train_test_split(train_pth, train_lab, test_size = valid_size)  
    
    # Returning TF's Dataset API
    train = tf.data.Dataset.from_tensor_slices((train_pth, train_lab))
    valid = tf.data.Dataset.from_tensor_slices((valid_pth, valid_lab))
    test = tf.data.Dataset.from_tensor_slices((test_pth, test_lab))

    train = train.map(load_img, num_parallel_calls=AUTO)\
        .map(one_hot, num_parallel_calls=AUTO)\
        .batch(BATCH_SIZE)\
        .map(Augment_data, num_parallel_calls=AUTO)\
        .cache()
    valid = valid.batch(BATCH_SIZE)
    test = test.batch(BATCH_SIZE)
    
    return train, valid, test
```


```
Epoch 1/5
2965/2965 [==============================] - 2460s 825ms/step - loss: 0.8089 - accuracy: 0.7714 - val_loss: 0.4552 - val_accuracy: 0.8682
Epoch 2/5
2965/2965 [==============================] - 2378s 802ms/step - loss: 0.5520 - accuracy: 0.8402 - val_loss: 0.3053 - val_accuracy: 0.9124
Epoch 3/5
2965/2965 [==============================] - 2371s 800ms/step - loss: 0.4185 - accuracy: 0.8772 - val_loss: 0.2379 - val_accuracy: 0.9310
Epoch 4/5
2965/2965 [==============================] - 2370s 799ms/step - loss: 0.3264 - accuracy: 0.9040 - val_loss: 0.1742 - val_accuracy: 0.9525
Epoch 5/5
2965/2965 [==============================] - 2383s 804ms/step - loss: 0.2577 - accuracy: 0.9256 - val_loss: 0.1122 - val_accuracy: 0.9718
```
