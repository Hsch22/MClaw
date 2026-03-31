# examples

`examples/` 目前只包含目标命令形状的说明性脚本。

## 当前文件

- `textcraft_train.sh`
  - 调用 `python -m mclaw.trainer.main`
  - 传入默认配置路径
  - 覆盖 `model.family` 和 `environment.adapter`
  - 继续透传额外命令行参数

## 当前状态

- 这个脚本目前是“将来接好 CLI 之后应该怎么调用”的示例。
- 由于 `mclaw.trainer.main.load_config()` / `build_trainer()` / `main()` 还没有实现，它暂时不是一个真正可运行的训练脚本。
