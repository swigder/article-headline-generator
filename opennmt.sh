DATA_DIR=../data/own
th tools/tokenize.lua < $DATA_DIR/training-samples.txt > $DATA_DIR/training-samples.txt.tok
th tools/tokenize.lua < $DATA_DIR/training-target.txt > $DATA_DIR/training-target.txt.tok
th tools/tokenize.lua < $DATA_DIR/validation-samples.txt > $DATA_DIR/validation-samples.txt.tok
th tools/tokenize.lua < $DATA_DIR/validation-target.txt > $DATA_DIR/validation-target.txt.tok
th preprocess.lua -train_src $DATA_DIR/training-samples.txt.tok -train_tgt $DATA_DIR/training-target.txt.tok -valid_src $DATA_DIR/validation-samples.txt.tok -valid_tgt $DATA_DIR/validation-target.txt.tok -src_seq_length 100 -tgt_seq_length 20 -save_data $DATA_DIR/own.t7-train.t7
th train.lua -data $DATA_DIR/own.t7-train.t7-train.t7 -save_model ../models/train.t7 -save_every_epochs 4 -gpuid 1


HOST=ec2-52-88-197-194.us-west-2.compute.amazonaws.com
scp -i ../oregon.pem.txt /Users/xx/Files/opennmt/data/own/*.tok ubuntu@$HOST:~/data

mkdir data
mkdir models
mkdir efs
sudo apt-get install nfs-common
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 fs-5b72a4f2.efs.us-west-2.amazonaws.com:/ efs
ln -s efs/ubuntu ~/models/efs
sudo nvidia-docker run -v ~/data:/var/data -v ~/models:/var/models -it harvardnlp/opennmt:8.0
cd
luarocks install tds
git clone https://github.com/OpenNMT/OpenNMT
cd OpenNMT
PREFIX=pos-src
th train.lua -data /var/data/$PREFIX-data-train.t7 -save_model /var/models/$PREFIX -save_every_epochs 4 -gpuid 1
th translate.lua -model /var/models/train.t7 -src /var/data/$PREFIX-training-samples.txt -tgt /var/data/$PREFIX-training-target.txt -gpuid 1


PREFIX=pos-src
SUFFIX=.txt
th preprocess.lua -train_src $DATA_DIR/$PREFIX-training-samples$SUFFIX -train_tgt $DATA_DIR/$PREFIX-training-target$SUFFIX -valid_src $DATA_DIR/$PREFIX-validation-samples$SUFFIX -valid_tgt $DATA_DIR/$PREFIX-validation-target$SUFFIX -src_seq_length 100 -tgt_seq_length 20 -save_data $DATA_DIR/$PREFIX-data
th train.lua -data /var/data/case-data-train.t7 -save_model /var/models/case.t7 -save_every_epochs 6 -gpuid 1