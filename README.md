# Open-XiaoAI x Sherpa-ONNX

> [!IMPORTANT]
> 本教程仅适用于 **小爱音箱 Pro（LX06）** 和 **Xiaomi 智能音箱 Pro（OH2P）**

小爱音箱自定义唤醒词，基于 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)。

## 演示视频

👉 [小爱音箱自定义唤醒词演示视频](https://www.bilibili.com/video/BV1YfVUz5EMj)

[![](https://github.com/idootop/open-xiaoai/raw/main/docs/images/kws.jpg)](https://www.bilibili.com/video/BV1YfVUz5EMj)

## 快速开始

> [!NOTE]
> 以下操作需要先将小爱音箱刷机， 然后 SSH 连接到小爱音箱。👉 [教程](https://github.com/idootop/open-xiaoai/blob/main/docs/flash.md)

```shell
# 创建 open-xiaoai/kws 文件夹
mkdir -p /data/open-xiaoai/kws

# 设置自定义唤醒词
cat <<EOF > /data/open-xiaoai/kws/keywords.txt
t iān m āo j īng l íng @天猫精灵
x iǎo d ù x iǎo d ù @小度小度
d òu b āo d òu b āo @豆包豆包
n ǐ h ǎo x iǎo zh ì @你好小智
EOF

# 设置唤醒提示语（可选）
cat <<EOF > /data/open-xiaoai/kws/reply.txt
主人你好，请问有什么吩咐？
EOF

# 运行启动脚本
curl -sSfL https://gitee.com/idootop/artifacts/releases/download/open-xiaoai-kws/init.sh | sh
```

如果你想要开机自启动，可以运行以下命令，然后重启小爱音箱即可。

```shell
# 下载到 /data/init.sh 开机时自启动
curl -L -o /data/init.sh https://gitee.com/idootop/artifacts/releases/download/open-xiaoai-kws/init.sh
```

## 设置唤醒词

在你的电脑上克隆本项目，然后修改 `scripts/my-keywords.txt` 文件里的唤醒词。注意：

- 仅支持中文唤醒词
- 支持多个唤醒词，每行一个

运行以下命令，更新 `scripts/keywords.txt` 里的唤醒词配置：

```shell
git clone https://github.com/idootop/sherpa-onnx.git

cd sherpa-onnx/scripts

# 你可能需要先安装 uv 👉 https://github.com/astral-sh/uv
uv run keywords.py --tokens tokens.txt --output keywords.txt --text my-keywords.txt
```

然后将你电脑上的 `scripts/keywords.txt` 复制到小爱音箱 `/data/open-xiaoai/kws/keywords.txt`。

如果你不方便复制文件，也可以直接在小爱音箱上运行以下命令（记得修改成自己的唤醒词）。

```shell
cat <<EOF > /data/open-xiaoai/kws/keywords.txt
t iān m āo j īng l íng @天猫精灵
x iǎo d ù x iǎo d ù @小度小度
d òu b āo d òu b āo @豆包豆包
n ǐ h ǎo x iǎo zh ì @你好小智
EOF
```

> [!TIP]
> 修改完毕后，记得重启脚本或小爱音箱使新配置生效。

## 设置欢迎语

你也可以设置自定义唤醒词唤醒后的提示语。首先新建一个 `reply.txt` 文件：

```txt
主人你好，请问有什么吩咐？
https://example.com/music.wav
file:///usr/share/sound-vendor/AiNiRobot/wakeup_ei_01.wav
```

注意：

- 每行一句欢迎语
- 支持文字和音频链接
- 多条欢迎语会随机选择一句播放
- 默认欢迎语与小爱同学一致：“哎”、“在”

然后将你电脑上的 `reply.txt`复制到小爱音箱 `/data/open-xiaoai/kws/reply.txt`。

如果你不方便复制文件，也可以直接在小爱音箱上运行以下命令（记得修改成自己的欢迎语）。

```shell
cat <<EOF > /data/open-xiaoai/kws/reply.txt
主人你好，请问有什么吩咐？
https://example.com/music.wav
file:///usr/share/sound-vendor/AiNiRobot/wakeup_ei_01.wav
EOF
```

> [!TIP]
> 修改完毕后，记得重启脚本或小爱音箱使新配置生效。

## 常见问题

### Q：为什么唤醒词识别不是很灵敏，时好时坏？

由于小爱音箱 client 端算力和存储空间有限，默认只使用了一个较小的语音识别模型，所以识别效果并不是很完美，需要多多尝试。

你可以在小爱音箱上运行以下调试脚本，根据语音识别结果，动态调整唤醒词拼音。

```shell
# 运行调试脚本
curl -sSfL https://gitee.com/idootop/artifacts/releases/download/open-xiaoai-kws/debug.sh | sh

# 🐢 模型加载较慢，请在提示 Started! Please speak 后，再使用自定义唤醒词
# Started! Please speak
# 0:tiānmāojīnglián 👈 天猫精灵
# 1:xiǎodùxiǎodù 👈 小度小度
# 2:dōubāodùba 👈 豆包豆包
# 3:nǐhǎoxiǎozhī 👈 你好小智
```

如果你想要更完美的唤醒词识别效果，可以在 server 端运行更大规模、更先进的 AI 模型，来进行唤醒词识别。推荐使用 [FunASR](https://github.com/modelscope/FunASR) 和 [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)，可以参考 [xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) 项目。

### Q：能不能设置英文/方言作为唤醒词？比如：Siri

默认使用的小型语音识别模型只支持中文（普通话）作为唤醒词。

如果你需要多语言唤醒词，比如英文、日语、韩语或方言等，可以在 server 端运行更大规模、更先进的 AI 模型，来进行唤醒词识别。推荐使用 [FunASR](https://github.com/modelscope/FunASR) 和 [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)，可以参考 [xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) 项目。

### Q：是否支持说话人（声纹）识别？

由于小爱音箱 client 端算力和存储空间有限，默认只使用了一个较小的语音识别模型，并不支持说话人识别等高阶功能。

如果你需要声纹识别、连续对话等高阶功能，可以在 server 端运行更大规模、更先进的 AI 模型，来进行语音识别和对话管理。推荐使用 [FunASR](https://github.com/modelscope/FunASR) 和 [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)，可以参考 [xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) 项目。

### Q：能否自定义唤醒词的响应动作？比如：关灯

该脚本默认不支持自定义唤醒词触发后执行的动作。

如果你想要自定义唤醒词的触发动作，可以参考 [open-xiaoai](https://github.com/idootop/open-xiaoai) 里的其他演示程序，根据 server 端接收到的小爱音箱上的唤醒词事件，实时触发任意自定义操作（比如：打电话，订外卖等）。
