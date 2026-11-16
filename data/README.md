# data/

Not tracked by git. `python scripts/get_data.py --agree` fills it:

```
data/raw/    sub1_comp.mat  sub2_comp.mat  sub3_comp.mat                    (BCI Competition IV, data set 4)
             sub1_testlabels.mat  sub2_testlabels.mat  sub3_testlabels.mat  (the true test labels)
data/prep/   sub1_prep.pt  sub2_prep.pt  sub3_prep.pt                       (python decoder/prep.py)
```

`python scripts/get_data.py --check` confirms the six raw files are the exact
copies the paper used (SHA-256).
