# Understanding the data format and classes

OpenScene is a compact redistribution of the large-scale [nuPlan dataset](https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/index.html), retaining only relevant annotations and sensor data at 2Hz. This reduces the dataset size by a factor of >10. The data used in NAVSIM is structured into `navsim.common.dataclasses.Scene` objects. A `Scene` is a list of `Frame` objects, each containing the required inputs and annotations for training a planning `Agent`.

**Filtering.** The NAVSIM validation an test sets will be filtered to increase the representation of challenging situations. Furthermore, the test set will only include agent inputs and exclude any privileged annotations.

**Caching.** Evaluating planners involves significant preprocessing of the raw annotation data, including accessing the global map at each ´Frame´ and converting it into a local coordinate system. You can generate the cache with:
```
cd $NAVSIM_DEVKIT_ROOT/scripts/
./run_metric_caching.sh
```

This will create the meric cache under `$NAVSIM_EXP_ROOT/metric_cache`, where `$NAVSIM_EXP_ROOT` is defined by the environment variable set during installation.
