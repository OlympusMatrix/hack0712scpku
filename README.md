- 训练
```shell
python train.py --epochs 30 --device cuda
```

- 验证
```shell
python val.py --weights results/exp5/weights/best.pt --device cuda:0
```

- 将输出结果转换为一个csv，作为submission
```shell
python test.py --weights /root/hackathon_2025_summer/results/exp5/weights/best.pt --device cuda:0 --save-txt
```
