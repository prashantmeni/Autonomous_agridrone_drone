# Training artifacts land here.

Created by the pipeline, not committed:

    model2.h5               best Keras model (training checkpoint)
    disease_model.onnx      exported ONNX for Raspberry Pi inference
    class_names.txt         38 labels, in index order
    label_mapping.json      dataset folder -> production index mapping
    metrics.json            machine-readable metrics
    evaluation.txt          human-readable metrics + confusion matrix
    training_config.yaml    config as actually used

The .gitignore in the project root should exclude *.onnx and *.h5.
Do not commit model weights without being asked explicitly.

**This directory is currently empty. No training run has been performed.**