# Music Generator

**這是一個使用了 GCN 與 LSTM 生成音樂旋律的模型。**

> 本模型參考源自作者 Halil Erdoğan 提出的LSTM音樂生成模型所改良。
來源自 : [https://hedonistrh.github.io/2018-04-27-Music-Generation-with-LSTM/](https://hedonistrh.github.io/2018-04-27-Music-Generation-with-LSTM/)

### 系統建議需求


|   | 最低配置  | 本實驗配置  |
| ------------ | ------------ | ------------ |
| OS  |  Win10 | Ubuntu 20.04 以上  |
| RAM  | 16G  | 64G |
| CPU  | 7th-i5(較無影響)  |  7th-i5 |
| GPU  | RTX 2070/8G  | RTX 3090/24G  |
| Cuda  | 依顯卡需求  | 11.7 |
&diams; 另外需安裝 [Musescore3](https://musescore.org/en/3.0 "Musescore3") 作為樂譜讀取和輸出用的載體。

### 使用說明
##### 資料預處理
訓練音樂前，需將音樂資料夾內的所有檔案 (.mxl 或 .mid) 打包成 .npy 以供後續程式做讀取。
因此需要先執行`/music_generation/preprocess_of_data.ipynb`，如下所示:
```python
def merge_list_and_tensor(list_data, tensor_data):
    
    list_data = torch.tensor(list_data, dtype=torch.float32)
    merge_list = torch.zeros((list_data.size(0), list_data.size(1)* 2))
    #merge_list = torch.zeros((list_data.size(0), 750))
	#若要取750張量長度，需先設置大小
    merge_list[:, :list_data.size(1)] = list_data
    merge_list[:tensor_data.size(0), list_data.size(1):] = tensor_data
	#merge_list = merge_list[:, :300]#take first 300
	#若要取長度為前300的張量長度
    return merge_list
```
```python
database_npy = 'your_dataset_name.npy' #資料集的名稱

print (os.getcwd())
root_dir = ('../Wikifonia_midi/') #音樂資料夾

all_midi_paths = glob.glob(os.path.join(root_dir, '*mxl')) #資料夾內音樂的格式

print (all_midi_paths)
matrix_of_all_midis = []
song_value = 50 #取多少首樂曲
count = 0 #僅作為計數器用，不需更改
BARS_LENGTH = 32 #設定樂曲長度至少為32個小節以上
```

------------


##### 音樂生成訓練
樂曲經過預處理接著便可執行`/music_generation/LSTM_run.ipynb`進行訓練,如下所示：
```python
environment.set('midiPath', '/usr/bin/musescore3')
#此為軟體Musescore3在Linux的位置,Win系統另需更改
```
```python
midis_array = './your_dataset_name.npy' #放入上述產生的資料集
midis_array_raw = np.load(midis_array)
#以下略
```
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.04, amsgrad=False,clipvalue=0.3) #可根據需求調整
model.compile(loss='categorical_crossentropy', optimizer=optimizer) #可根據需求調整
```
```python
from keras import layers
from keras import models
import keras
from keras.models import Model
import tensorflow as tf
from keras.layers.advanced_activations import *
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 設置使用顯示卡顯存的比例，例如0.5表示使用一半的顯存，不需要限制可刪除此行
session = tf.compat.v1.Session(config=config)
#以下略
```
```python
import random
import sys

epoch_total = 801 #可根據需求設置
batch_size = 128 #可根據需求設置

#以下略

          generated_midi_final = np.transpose(generated_midi,(1,0))
          output_notes = matrix_to_midi(generated_midi_final, random=1)
          midi_stream = stream.Stream(output_notes)
          midi_file_name = ('lstm_out_{}_{}.mid'.format(epoch, temperature))
          midi_stream.write('midi', fp=midi_file_name)
          parsed = converter.parse(midi_file_name)
          for part in parsed.parts:
              part.insert(0, instrument.Piano())
          parsed.write('midi', fp=midi_file_name) #輸出為midi的樂曲
```
```python
import bottleneck 
z = -bottleneck.partition(-preds, 20)[:20]
print (z)
print ('max:', np.max(preds))

model.save('my_model.h5') #儲存訓練完成的模型資料

model.save_weights('my_model_weights.h5')
```

------------

##### 音樂生成
訓練完成後，執行`/music_generation/Music_generator.ipynb`生成兩首音樂，如下所示:
```python
midis_array = './your_dataset_name.npy' #同樣更換資料集
#midis_array = './midis_array_wikifonia.npy'
#midis_array = './midis_array_wikifonia_200.npy'
midis_array_raw = np.load(midis_array)
#以下略
```
```python
import random
from keras.models import load_model
import numpy as np

model = Model(input_midi, x)
model.load_weights('./my_model_weights.h5') #訓練後的模型資料
#以下略
```

------------

##### 圖表資料
因本實驗僅取樂曲當中小節之間的兩種關係(重複關係、節奏順序關係)來生成音樂，可執行參考`/music_generation/Music_generator.ipynb`內的鄰接矩陣圖，如下所示:
```python
music_file = './your_music.mxl' #放入需要檢視的樂曲

BARS_LENGTH = 32 #擷取長度為32小節

bars_repeat, bars_rhythm = music_to_adjacency_metrix(music_file, BARS_LENGTH)

visualize_adjacency_matrix(bars_repeat)
visualize_adjacency_matrix(bars_rhythm)
```

&diams; 資料集可使用 [POP909](https://github.com/music-x-lab/POP909-Dataset)，[Wikifonia](https://github.com/00sapo/OpenEWLD) 以及原作者使用的 [Schumann](https://github.com/hedonistrh/bestekar) 來測試。

詳細的程式碼等資料請參閱我的 [Github](https://github.com/Zedekian "Github") 