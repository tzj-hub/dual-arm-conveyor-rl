# 第三方来源与许可证

## Franka Emika Panda 模型

- 上游项目：[Google DeepMind MuJoCo Menagerie / franka_emika_panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)。
- 本工程模型位置：`src/dual_arm_conveyor/assets/robots/panda/`。
- 许可证：Apache License 2.0，原文随模型保存在 `LICENSE`。许可证于 2026-11-02 从上游公开地址 `https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/franka_emika_panda/LICENSE` 获取，未修改正文。
- 模型与网格复制自用户已有实习工程 `src/dual_arm_sort_mjlab/resources/assets/robots/panda/`，而非重新下载当前上游版本。原工程未记录对应上游 commit，因此本工程不宣称所含 XML 与当前上游完全一致。
- 继承的 XML 已有接触参数、末端 site 和腕部相机相关配置。本工程保留这些本地版本文件；`model.py` 在构建场景时添加两臂命名空间、设置基座位姿、调整执行器参数并移除未使用的腕部相机配置。运行时场景由 Python 生成，网格几何未重新建模。
- v6 在启用腕部模式时，额外生成两套可见的相机支架、外壳、镜头与 RGB-D 相机，并使用重新验证的手眼安装变换；这些安装件目前无质量、无碰撞。原资产文件中的相机安装配置不直接参与运行，Panda 视觉与碰撞网格保持原样。
- 运行时将每指原有的五块重叠指垫碰撞盒替换为同一外包络内的一块平面指垫，并将指部接触摩擦参数设为 `1.0 / 0.015 / 0.001`，以消除小台阶导致的圆柱开指后卡滞。该修改仅作用于生成的碰撞基本体；视觉网格和原始网格几何保持不变。模型目录中的 `NOTICE` 同步记录了这些修改。

Apache-2.0 的许可证范围仍适用于上游模型部分；本文不更改第三方授权条件。仓库原创代码采用根目录 `LICENSE` 中的 MIT 许可证；该许可证不替代模型或下述 MuJoCo 适配代码的 Apache-2.0 许可。

## 运行依赖

MuJoCo、NumPy、SciPy、PyTorch、Matplotlib、imageio、Pillow 和 pytest 通过各自的 Python 分发包安装，除下述深度换算适配外，本工程没有合并这些依赖的源码。各依赖保留其分发包附带的许可证。依赖版本范围见 `pyproject.toml`；测试和训练使用的实际版本记录在结果与运行检查输出中。

## MuJoCo RGB-D 深度换算

`src/dual_arm_conveyor/rgbd.py` 的 reverse-Z 深度还原适配自 [MuJoCo 3.17.0 classic renderer](https://github.com/google-deepmind/mujoco/blob/3.17.0/python/mujoco/rendering/classic/renderer.py)，Copyright 2022 DeepMind Technologies Limited，Apache-2.0。本工程增加了场景与版本检查、RGB 深度缓冲复用和公开 API 回退；未改变深度投影换算顺序。原版权及许可证声明保留在模块中，许可证全文位于 `src/dual_arm_conveyor/assets/licenses/MUJOCO_LICENSE`，随 wheel 一同打包。
