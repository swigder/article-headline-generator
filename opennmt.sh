DATA_DIR=../data/own
th tools/tokenize.lua < $DATA_DIR/training-samples.txt > $DATA_DIR/training-samples.txt.tok
th tools/tokenize.lua < $DATA_DIR/training-target.txt > $DATA_DIR/training-target.txt.tok
th tools/tokenize.lua < $DATA_DIR/validation-samples.txt > $DATA_DIR/validation-samples.txt.tok
th tools/tokenize.lua < $DATA_DIR/validation-target.txt > $DATA_DIR/validation-target.txt.tok
th preprocess.lua -train_src $DATA_DIR/training-samples.txt.tok -train_tgt $DATA_DIR/training-target.txt.tok -valid_src $DATA_DIR/validation-samples.txt.tok -valid_tgt $DATA_DIR/validation-target.txt.tok -src_seq_length 100 -tgt_seq_length 20 -save_data $DATA_DIR/own.t7-train.t7
th train.lua -data $DATA_DIR/own.t7-train.t7-train.t7 -save_model ../data/train.t7 -save_every 200 -gpuid 1