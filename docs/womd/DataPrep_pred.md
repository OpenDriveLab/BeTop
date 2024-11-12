# Preparation for WOMD prediction

## Installation

Run the following command to install the BeTopNet (WOMD) package:

```bash
cd womd && pip install -e .
```

To build the necessary CUDA dependencies, please refer to [EQNet](https://github.com/dvlab-research/DeepVision3D/tree/master/EQNet/eqnet/ops).

Make sure the corresponding WOMD package are successfully installed

## Data Preparation

### 1. Download the Data

Download the ```scenario/``` part of Waymo Open Motion Dataset under certain path and your specified task: [Link](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_3_0;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)

Download the ```intention_points.pkl```. You may find the corresponding files in the following [Link](https://github.com/OpenDriveLab/BeTop/releases/tag/womd)

### 2. Preprocess the Raw Data:

**[NOTE]** The ```data_process``` code assumes you have downloaded both motion & interaction data, so you may modify according to your need.

Run the following command to preprocess the raw data:

```bash
cd tools/data_tools
python3 data_process.py your_downloaded_path your_preferred_info_path
```

### 3. **[Optional]** Cache Preprocessed Data

For memory-efficient training, you can preprocess the info data in ```.npz``` format cache:

```bash
cd tools/data_tools
python3 cache_offline_data.py --cache_path YOUR_CACHE_PATH --cfg ../cfg/BeTopNet_full_64.yaml
```

**[NOTE]** The caching process would require around ```3-4TB``` storage space.