# MPIIGaze　データセットを用いた2D視線推定モデル
## GPU必須

## 前処理
・MPIIGaze、MPIIFaceGazeを用意

```
python my_data_preprocess.py
```



foldを指定してトレーニング
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