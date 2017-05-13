# on your machine, to preprocess data (that you have output using create_training_files)
DATA_DIR=../data/own # MODIFY FOR YOUR OWN LOCATION
PREFIX='' # ENTER YOUR PREFIX HERE
th preprocess.lua -train_src $DATA_DIR/$PREFIX-training-samples.txt -train_tgt $DATA_DIR/$PREFIX-training-target.txt -valid_src $DATA_DIR/$PREFIX-validation-samples.txt -valid_tgt $DATA_DIR/$PREFIX-validation-target.txt -src_seq_length 100 -tgt_seq_length 20 -save_data $DATA_DIR/$PREFIX

# get a spot request from AWS
# AMI:  (Oregon)
# Machine type: p2.xlarge
# Spot price: .3 usually works

KEY_FILE=../oregon.pem.txt # MODIFY FOR YOUR OWN KEYFILE

# to move your data to the AWS machine
HOST='' # ENTER THE MACHINE HERE
scp -i $KEY_FILE $DATA_DIR/$PREFIX-data-train.t7 ubuntu@$HOST:~/data

# to log on to the AWS machine
ssh -i $KEY_FILE ubuntu@$HOST

# on the AWS machine
# skip these if using our AMI
mkdir data
mkdir models
mkdir efs
sudo apt-get install nfs-common
# skip this if already mounted
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 fs-5b72a4f2.efs.us-west-2.amazonaws.com:/ efs
# start docker
sudo nvidia-docker run -v ~/data:/var/data -v ~/models:/var/models -v ~/efs/ubuntu:/var/efs -it harvardnlp/opennmt:8.0

# in docker
# setup
cd
luarocks install tds
git clone https://github.com/OpenNMT/OpenNMT
cd OpenNMT
# train - set your own training parameters
PREFIX='' # ENTER YOUR PREFIX HERE
th train.lua -data /var/data/$PREFIX-data-train.t7 -save_model /var/models/$PREFIX -save_every_epochs 4 -gpuid 1
# translate - update the location of the model
th translate.lua -model /var/models/${PREFIX}_epoch12*.t7 -src /var/data/$PREFIX-test-samples.txt -tgt /var/data/$PREFIX-test-target.txt -gpuid 1
# save to efs
mkdir /var/efs/$PREFIX
mv /var/models/${PREFIX}_epoch12*.t7 /var/efs/$PREFIX
