# Submission for WOMD Leaderboard

## Submission Info

Specify your registered information in submission code:

1. Trace to Line 188 in ```womd/tools/submission.py```.
2. Specify your information in the following format:

```python
submission_info = dict(
        account_name='your Waymo account email',
        unique_method_name='xxx',
        authors=['A', 'B', 'xxx'],
        affiliation='your affiliation',
        uses_lidar_data=False,
        uses_camera_data=False,
        uses_public_model_pretraining=False,
        public_model_names='N/A',
        num_model_parameters='N/A',
    )
```


## Submission File Generation

Run the submission pipeline:

```bash
cd womd/tools
python3 test_submission.py \
        --cfg_file cfg/YOUR_CHOSEN_BASELINE.yaml \
        --batch_size BATCH \
        --ckpt MODEL_CKPT \
        --output_dir path_to_save_submission_file \
        #--eval \ # whether to submit the eval / test portal
        #--interactive # whether to submit the motion / interaction prediction task
```

**Note:** Fot the last two arguments, the default setup is for ```test``` submission for *Motion Prediction* challenge.

- Use ```--eval``` if you want to check the evaluation on the Leaderboard portal.
- Add the ```--interactive``` argument if you want to submit for *Interaction Prediction* challenge.

## Upload

Submit your results to the Leaderboard portal:

You can find the corresponding ```.tar.gz``` file under your specified ```--output_dir```. Submit to the Leaderboard portal:

- [Motion Prediction challenge](https://waymo.com/open/challenges/2024/motion-prediction/)
- [Interaction Prediction challenge](https://waymo.com/open/challenges/2021/interaction-prediction/)