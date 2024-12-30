# 双臂移动机器人项目

## 项目简介

本项目旨在开发一个双臂移动机器人，集成语音、视觉与运动控制模块，实现复杂场景中的分拣与抓取任务。机器人将在医疗场景下展示其能力，例如试管、口罩、手套等物品的自动化分拣与处理。

---

## 项目结构

```
DualArm_Robot_Project/
├── docs/                  # 文档与说明
├── hardware/              # 硬件设计与集成
│   ├── mechanical_design/ # 机械设计
│   ├── sensors/           # 传感器选型与调试
│   └── controllers/       # 控制器配置
├── software/              # 软件开发
│   ├── vision/            # 视觉模块（YOLO, 点云处理）
│   ├── speech/            # 语音模块（Whisper, ChatTTS）
│   ├── motion_control/    # 运动控制
│   └── integration/       # 系统集成代码
├── tests/                 # 测试用例和日志
└── exhibits/              # 展示相关内容（场景配置、演示视频等）
```

---

## 主要功能

1. **语音交互**：
   - 集成 Whisper 模型进行语音转文字。
   - 使用大语言模型（如 LLaMA）解析用户指令。
   - 支持实时语音响应。

2. **视觉识别**：
   - YOLO 模型用于目标检测（试管、口罩等）。
   - 结合深度摄像头生成点云数据。

3. **运动控制**：
   - 精确的机械臂控制。
   - 动态路径规划与抓取优化。

4. **系统集成**：
   - 集成语音、视觉与运动控制模块。
   - 支持复杂任务的自动化执行。

---

## 环境依赖

### 硬件依赖
- **机械臂**：协作机器人或工业机械臂。
- **摄像头**：Intel D435i、Azure Kinect DK。
- **其他传感器**：IMU、激光传感器等。

### 软件依赖
- Python >= 3.8
- Conda 环境：
  - 主环境：`dualarm_env`
  - 视觉模块：`dualarm_vision`
  - 语音模块：`dualarm_speech`
  - 运动控制模块：`dualarm_motion`
- 主要库：
  - `torch`
  - `transformers`
  - `open3d`
  - `yolov5`

---

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/your-repo/DualArm_Robot_Project.git
cd DualArm_Robot_Project
```

### 2. 创建环境
使用 Conda 创建所需环境：
```bash
conda create -n dualarm_env python=3.8
conda activate dualarm_env
# 安装依赖库
pip install -r requirements.txt
```

### 3. 运行测试
进入测试目录运行测试代码：
```bash
cd tests
python test_robot.py
```

---

## 贡献指南

欢迎大家为本项目贡献代码或提出改进意见。

1. 提交问题 (Issue)：通过 GitHub 提交。
2. 拉取请求 (Pull Request)：
   - 确保代码格式符合项目规范。
   - 在提交之前运行相关测试。

---

## 许可证

本项目使用 [MIT License](LICENSE)。

