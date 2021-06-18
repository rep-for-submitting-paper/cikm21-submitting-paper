# cikm21-submitting-paper
The code of submitting paper. submission id 2958


### run scripts

step1: construct knn-graph
```
python3 evoluNetwork.py
```

step2: train and test model
```
python3 tarin.py --view app_drebin_app --keyword drebin
```


The dataset used in this repo comes from [TESSERACT: eliminating experimental bias in malware classification across space and time.](https://dl.acm.org/doi/abs/10.5555/3361338.3361389). Make sure dataset is saved and preprocessed in correspondding dir configed by `setting.py` before runing scripts.