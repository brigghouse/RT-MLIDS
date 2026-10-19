# Data

The experiments use **NSL-KDD** only. The files are not redistributed in this repository (`data/NSL_KDD/` is git-ignored).

- **Source:** [Canadian Institute for Cybersecurity — NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
- **Files needed:** `KDDTrain+.txt` (125,973 records) and `KDDTest+.txt` (22,544 records)
- **Raw features:** 41 (plus label and difficulty columns); 3 categorical (`protocol_type`, `service`, `flag`) become 122 one-hot columns across both files.
- **Classes:** normal, DoS, Probe, R2L, U2R. KDDTest+ contains attack types that never appear in KDDTrain+ (e.g. `mscan`, `apache2`, `snmpguess`, `mailbomb`), which is why accuracy on it is far lower than on random splits.

```bash
mkdir -p data/NSL_KDD
# copy KDDTrain+.txt and KDDTest+.txt into data/NSL_KDD/
python rt_mlids_experiment.py --data-path data/NSL_KDD
```

CIC-IDS-2018 is **not** used in this project; evaluation on it is listed as future work.
