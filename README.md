# MPIIGaze　データセットを用いた2D視線推定モデル





```
python train_v9.py --fold XX
```



各foldのベスト検証スコアのみでテスト
```
python eval_v9.py --onlybest False
```

各foldの全てのcheckpointでテスト
```
python eval_v9.py --onlybest False
```