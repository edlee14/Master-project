import pandas as pd
import numpy as np
import random
import sys
import os
import h5py
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tldextract
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import regex as re
from typing import *


from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer

from keras.models import Sequential
from keras.layers import Dropout,LSTM,Embedding, Flatten, Dense
from keras.preprocessing import sequence
from keras import layers
from keras import optimizers
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

import pydot
import pydotplus
import keras.utils.vis_utils
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
from pydot import *
from graphviz import *

pd.set_option('display.width',None)
url_data_all = pd.read_csv('C:/Users/lh14532/phishing_site_urls.csv')
url_data_all = url_data_all.sample(frac=1).reset_index(drop=True)
url_data = url_data_all.iloc[:100]
#url_data = url_data_all
print(url_data)
url_data = url_data.sample(frac=1).reset_index(drop=True)
print(url_data)
# pd.set_option('display.max_colwidth', -1)




url_data.sample(10)

url_data = url_data.rename(columns={"URL": "url", "Label": "label"})


def parse_url(url: str) -> Optional[Dict[str, str]]:
    try:
        no_scheme = not url.startswith('https://') and not url.startswith('http://')
        if no_scheme:
            parsed_url = urlparse(f"http://{url}")
            return {
                "scheme": None,  # not established a value for this
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
        else:
            parsed_url = urlparse(url)
            return {
                "scheme": parsed_url.scheme,
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
    except:
        return None

print(url_data)
print('Preparing ')

url_data["parsed_url"] = url_data.url.apply(parse_url)

url_data = pd.concat([
    url_data.drop(['parsed_url'], axis=1),
    url_data['parsed_url'].apply(pd.Series)
], axis=1)

print(url_data)

url_data = url_data[~url_data.netloc.isnull()]

url_data["length"] = url_data.url.str.len()

url_data["tld"] = url_data.netloc.apply(lambda nl: tldextract.extract(nl).suffix)
url_data['tld'] = url_data['tld'].replace('', 'None')


url_data["is_ip"] = url_data.netloc.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")
url_data["is_ip"].dtype
url_data["is_ip"].astype(np.int64)
url_data["is_ip"] = url_data["is_ip"].astype(np.int64)
url_data["is_ip"].dtype


print(url_data)

url_data['domain_hyphens'] = url_data.netloc.str.count('-')
url_data['domain_underscores'] = url_data.netloc.str.count('_')
url_data['path_hyphens'] = url_data.path.str.count('-')
url_data['path_underscores'] = url_data.path.str.count('_')
url_data['slashes'] = url_data.path.str.count('/')

url_data['full_stops'] = url_data.path.str.count('.')

print(url_data)

def get_num_subdomains(netloc: str) -> int:
    subdomain = tldextract.extract(netloc).subdomain
    if subdomain == "":
        return 0
    return subdomain.count('.') + 1


url_data['num_subdomains'] = url_data['netloc'].apply(lambda net: get_num_subdomains(net))

tokenizer = RegexpTokenizer(r'[A-Za-z]+')

print(url_data)

print('tokenizing')
def tokenize_domain(netloc: str) -> str:
    split_domain = tldextract.extract(netloc)
    no_tld = str(split_domain.subdomain + '.' + split_domain.domain)
    return " ".join(map(str, tokenizer.tokenize(no_tld)))


url_data['domain_tokens'] = url_data['netloc'].apply(lambda net: tokenize_domain(net))

url_data['path_tokens'] = url_data['path'].apply(lambda path: " ".join(map(str, tokenizer.tokenize(path))))

url_data["label"].str.len()
url_data["target"] = url_data["label"].str.len()
url_data["target"] = url_data["target"]- 3
print(url_data["target"])

print(url_data)
print('remove useless')
#url_data_y = url_data['label']
url_data.drop('label', axis=1, inplace=True)
url_data.drop('url', axis=1, inplace=True)
url_data.drop('scheme', axis=1, inplace=True)
url_data.drop('netloc', axis=1, inplace=True)
url_data.drop('path', axis=1, inplace=True)
url_data.drop('params', axis=1, inplace=True)
url_data.drop('query', axis=1, inplace=True)
url_data.drop('fragment', axis=1, inplace=True)

#url_data["label"].dtype
#url_data["label"].astype(np.int64)
#url_data["label"] = url_data["label"].astype(np.int64)
#url_data["label"].dtype



print(url_data)
print('break point1')


url_data_train = url_data.sample(frac=0.8, random_state=1444)
url_data_mid = url_data.drop(url_data_train.index)
url_data_mid = url_data_mid.sample(frac=1).reset_index(drop=True)

url_data_val = url_data_mid.sample(frac=0.5, random_state=1444)
print('val:', url_data_val)
url_data_test = url_data_mid.drop(url_data_val.index)
print('test', url_data_test)

url_data_train_y = url_data_train["target"]
url_data_val_y = url_data_val["target"]
url_data_test_y = url_data_test["target"]

#def dataframe_to_dataset_y(url_data):
#    targets = url_data.copy()
#    ds = tf.data.Dataset.from_tensor_slices(targets)
#    return ds
#val_ds_y = dataframe_to_dataset_y(url_data_val_y)
#train_ds_y = dataframe_to_dataset_y(url_data_train_y)
#test_ds_y = dataframe_to_dataset_y(url_data_test_y)



def dataframe_to_dataset(url_data):
    url_data = url_data.copy()
    targets = url_data.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(url_data), targets))
    return ds
print(
    "Using %d samples for training and %d for validation"
    % (len(url_data_train), len(url_data_val))
    )
train_ds = dataframe_to_dataset(url_data_train)
val_ds = dataframe_to_dataset(url_data_val)
test_ds = dataframe_to_dataset(url_data_test)



for x, y in train_ds.take(1):
    print("Input:", x)
    print("Label:", y)
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)
test_ds = test_ds.batch(32)
print('building...')


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

def encode_string_categorical_feature(feature, name, dataset):
    # Create a StringLookup layer which will turn strings into integer indices
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):
    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(feature)
    return encoded_feature


tld = keras.Input(shape=(1,), name="tld", dtype="string")
domain_tokens = keras.Input(shape=(1,), name="domain_tokens", dtype="string")
path_tokens = keras.Input(shape=(1,), name="path_tokens", dtype="string")


is_ip = keras.Input(shape=(1,), name="is_ip", dtype="int64")
length = keras.Input(shape=(1,), name="length", dtype="int64")
domain_hyphens = keras.Input(shape=(1,), name="domain_hyphens", dtype="int64")
domain_underscores = keras.Input(shape=(1,), name="domain_underscores", dtype="int64")
path_hyphens = keras.Input(shape=(1,), name="path_hyphens", dtype="int64")
path_underscores = keras.Input(shape=(1,), name="path_underscores", dtype="int64")
slashes = keras.Input(shape=(1,), name="slashes", dtype="int64")
full_stops = keras.Input(shape=(1,), name="full_stops", dtype="int64")
num_subdomains = keras.Input(shape=(1,), name="num_subdomains", dtype="int64")



model = Sequential()
all_inputs = [
    is_ip,
    length,
    domain_hyphens,
    domain_underscores,
    path_hyphens,
    path_underscores,
    slashes,
    full_stops,
    num_subdomains,
    tld,
    domain_tokens,
    path_tokens,

]


# String categorical features
tld_encoded = encode_string_categorical_feature(tld, "tld", train_ds)
domain_tokens_encoded = encode_string_categorical_feature(domain_tokens, "domain_tokens", train_ds)
path_tokens_encoded = encode_string_categorical_feature(path_tokens, "path_tokens", train_ds)

#Integer categorical features

is_ip_encoded = encode_integer_categorical_feature(is_ip, "is_ip", train_ds)

#Numerical features
length_encoded = encode_numerical_feature(length, "length", train_ds)
domain_hyphens_encoded = encode_numerical_feature(domain_hyphens, "domain_hyphens", train_ds)
domain_underscores_encoded = encode_numerical_feature(domain_underscores, "domain_underscores", train_ds)
path_hyphens_encoded = encode_numerical_feature(path_hyphens, "path_hyphens", train_ds)
path_underscores_encoded = encode_numerical_feature(path_underscores, "path_underscores", train_ds)
slashes_encoded = encode_numerical_feature(slashes, "slashes", train_ds)
full_stops_encoded = encode_numerical_feature(full_stops, "full_stops", train_ds)
num_subdomains_encoded = encode_numerical_feature(num_subdomains, "num_subdomains", train_ds)

print('encoding')






print('layer')

all_features = layers.concatenate(
    [
        tld_encoded,
        domain_tokens_encoded,
        path_tokens_encoded,
        
        is_ip_encoded,
        
        length_encoded,
        domain_hyphens_encoded,
        domain_underscores_encoded,
        path_hyphens_encoded,
        path_underscores_encoded,
        slashes_encoded,
        full_stops_encoded,
        num_subdomains_encoded,
        
    ]
)
print('layer begin')
x = layers.Dropout(0.8)(all_features)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(16,activation = 'relu')(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)





















()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics = ['accuracy'])
print('Compiled!')
#keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
history = model.fit(
                train_ds,
                epochs=10,
                validation_data = val_ds
                )
                     
    

print('model')


#plot_model(model, show_shapes=True,to_file='model.png' , rankdir="LR")




y_pred = model.evaluate(test_ds,  verbose = 1)
print(y_pred)
history_dict = history.history
history_dict.keys()
loss, accuracy = model.evaluate(test_ds)
print('test loss: ', loss)
print('test accuracy: ', accuracy)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
epochs = range(1, len(acc) + 1)
#Plot Loss
print('plot loss......')
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
#Plot Accuracy
print('plot accuracy......')
plt.plot(epochs, acc_values, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,#set color for the matrix
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))


predictions = model.predict(test_ds)
for index in range(len(predictions)):
  if predictions[index] > 0.500 :
    predictions[index] = 1
  else:
    predictions[index] = 0



def plot_confuse(model, url_data_test,url_data_test_y,predictions):
    
    truelabel = url_data_test_y   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title='Confusion Matrix')
labels=['0','1']

plot_confuse(model, url_data_test,url_data_test_y,predictions)

print('start plot PR')
y_true = url_data_val_y
x_scores = predictions

p, r, thresholds = precision_recall_curve(y_true, x_scores)
plt.figure(1)
plt.plot(p, r)
plt.legend()
plt.show()
print('plot over')

sample = {
    "is_ip":1 ,
    "length":1 ,
    "domain_hyphens": 1 ,
    "domain_underscores": 1 ,
    "path_hyphens":1  ,
    "path_underscores": 1 ,
    "slashes":1  ,
    "full_stops": 1 ,
    "num_subdomains": 1 ,
    "tld": 1  ,
    "domain_tokens": 1  ,
    "path_tokens":1   

}
#input_dict = { name: tf.covert_to_tensor([value]) } for name, value in sample.items()}
#The_predictions = model.predict(input_dict)
#print(
#   "This particular patient had a %.1f percent probability "
#    "of having a heart disease, as evaluated by our model." % (100 * predictions[0][0],) 
#)
