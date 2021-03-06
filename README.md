# dialogGenerator_transformer

## Installation
```bash
pip install -r requirement.txt
```

## 학습 방법

train.py / train_key.py 파일 실행 (키워드의 사용 유무)

train의 argument:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset.
keyword_module | `str` | `""` | Use keyword module or not
train_batch_size | `int` | `20` | Batch size for training
valid_batch_size | `int` | `20` | Batch size for validation
gradient_accumulation_steps | `int` | `8` | Accumulate gradients on several steps
lr | `float` | `6.25e-5` | Learning rate
max_norm | `float` | `1.0` | Clipping gradient norm
n_epochs | `int` | `5` | Number of training epochs
personality_permutations | `int` | `1` | Number of permutations of personality sentences
device | `str` | `"cuda" if torch.cuda.is_available() else "cpu"` | Device (cuda or cpu)
fp16 | `str` | `""` | Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)
local_rank | `int` | `-1` | Local rank for distributed training (-1: not distributed)
gpt2_model_name | `str` | `"gpt2"` | Path, url or short name of the model

 
## 문장 생성 방법

interact.py / interact_key.py 파일 실행 (키워드의 사용 유무)

interact의 argument:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset.
model_checkpoint | `str` | `""` | Path, url or short name of the model
device | `str` | `cuda` if `torch.cuda.is_available()` else `cpu` | Device (cuda or cpu)
gpt2_model_name | `str` | `"gpt2"` | name of the model ex)openai-gpt
no_sample | action `store_true` | Set to use greedy decoding instead of sampling
max_length | `int` | `40` | Maximum length of the output utterances
min_length | `int` | `1` | Minimum length of the output utterances
seed | `int` | `0` | Seed
temperature | `int` | `0.7` | Sampling softmax temperature
top_k | `int` | `0` | Filter top-k tokens before sampling (`<=0`: no filtering)
top_p | `float` | `0.9` | Nucleus filtering (top-p) before sampling (`<=0.0`: no filtering)

```bash
python interact.py --dataset_path DATAPATH/Name --model_checkpoint MODELPATH/
python interact_key.py --dataset_path DATAPATH/Name --model_checkpoint MODELPATH/
```

## 데이터 포맷

Source|Target 형태로 txt파일 구성.

아래의 형태로 같은 경로에 데이터가 존재해야 함.  
Name_train.txt  / Name_train_keyword.txt  
Name_valid.txt  / Name_valid_keyword.txt  
Name_test.txt   / Name_test_keyword.txt  

## Reference
@article{DBLP:journals/corr/abs-1901-08149,
  author    = {Thomas Wolf and
               Victor Sanh and
               Julien Chaumond and
               Clement Delangue},
  title     = {TransferTransfo: {A} Transfer Learning Approach for Neural Network
               Based Conversational Agents},
  journal   = {CoRR},
  volume    = {abs/1901.08149},
  year      = {2019},
  url       = {http://arxiv.org/abs/1901.08149},
  archivePrefix = {arXiv},
  eprint    = {1901.08149},
  timestamp = {Sat, 02 Feb 2019 16:56:00 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1901-08149},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
