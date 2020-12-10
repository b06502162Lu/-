#!/bin/bash

gdown --id 13gxXY5_Khh9Rx4qIZyZ8C13CYzVPoNnx
gdown --id 1uVcw3TjcBguXDS7DnpzNqnUb8IiFr6rJ
gdown --id 1kGPYna0WwWRvHGQoyHnX3m5_MOg3SNft

python3 model1_train.py $1 $2

python3 model2_train.py $1 $2

python3 model3_train.py $1 $2

python3 model4_train.py $1 $2

python3 model5_train.py $1 $2

python3 model6_train.py $1 $2

python3 model7_train.py $1 $2

python3 model8_train.py $1 $2

python3 model9_train.py $1 $2

python3 model10_train.py $1 $2