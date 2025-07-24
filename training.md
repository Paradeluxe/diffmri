
## 2. 数据预处理

使用以下命令对音频数据进行预处理，生成梅尔频谱图：

```bash
python -m diffwave.preprocess /path/to/dir/containing/wavs
```

这将在指定目录中生成`.spec.npy`文件。

## 3. 模型训练

通过以下命令启动模型训练：

```bash
python -m diffwave /path/to/model/dir /path/to/dir/containing/wavs
```

其中：
- `/path/to/model/dir` 是模型检查点和日志存储目录。
- `/path/to/dir/containing/wavs` 是包含`.wav`文件的数据目录。

### 多GPU训练

DiffWave支持多GPU训练，可以显著加快训练速度。使用以下命令启动多GPU训练：

```bash
python -m torch.distributed.launch --nproc_per_node=<number_of_gpus> -m diffwave /path/to/model/dir /path/to/dir/containing/wavs
```

## 4. 监控训练进度

使用TensorBoard监控训练进度：

```bash
tensorboard --logdir /path/to/model/dir --bind_all
```

## 5. 调整训练参数

你可以通过修改`src/diffwave/params.py`文件来调整训练参数，例如：
- `batch_size`
- `learning_rate`
- `sample_rate`
- `n_mels`

这些参数将影响模型的训练效果和速度。

## 注意事项

- 确保音频文件为16位单声道`.wav`格式。
- 预处理和训练时指定正确的目录路径。
- 根据硬件配置调整`batch_size`等参数以优化训练性能。