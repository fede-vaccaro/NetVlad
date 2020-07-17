# first, do 'export MODEL='path_to_model'
echo "Training PCA from " $MODEL "on GPU: " $GPU "from configuration: " $CONF
# python train_pca.py -m $MODEL -c $CONF -d $GPU
python extract_features_revisitop.py -m $MODEL -d $GPU -c $CONF -o && python extract_features_revisitop.py -m $MODEL -d $GPU -c $CONF -p
python example_evaluate.py -o && python example_evaluate.py -p
