# Training & Evaluation for WOMD prediction

## Baseline Model Selection
Choose a baseline model to train through config files in ```womd/tools/cfg/```:

```bash
BeTopNet-full: cfg/BeTopNet_full_64.yaml
BeTopNet-e2e: cfg/BeTopNet_e2e_6.yaml
MTR++: cfg/MTR_PlusPlus.yaml
Wayformer: cfg/Wayformer.yaml
```

## Training
To start the training pipeline, run the following command:

```bash
cd womd/tools
bash scripts/dist_train.sh N_GPUS --cfg_file cfg/YOUR_CHOSEN_BASELINE.yaml --epoch 30 --batch_size BATCH --extra_tag XXX
```

**[TIPS]** If you have not cached the data, your may set ```BATCH = 10*N_GPUS``` using A100(80G) gpus. Otherwise, you may leverage a larger batch size.

## Evaluation
To evaluate the trained model, use the following command:

```bash
cd womd/tools
bash scripts/dist_test.sh N_GPUS --cfg_file cfg/YOUR_CHOSEN_BASELINE.yaml --batch_size BATCH --ckpt YOUR_MODEL_CKPT
```
