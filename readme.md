

# Funsound 多路语音识别
<img src="./img/logo.png" alt="Funsound Logo" width="100" height="100">

官网：[https://www.funsound.cn](https://www.funsound.cn)

魔塔：https://modelscope.cn/studios/QuadraV/FunSound

## 1. 特性

Funsound 整合了 **Funasr**, **Whisper**, 和 **Sherpa** 等开源语音识别方案，构建了一个高效的语音识别部署系统。Funsound2 基于 **WebSocket** 协议进行语音上传和实时结果返回。

### 主要特性：
- 面向多路的异步并发服务端设计
- 继承 **Whisper** 和 **Funasr** 等 **SOTA**（state-of-the-art）开源模型
- 支持 **时间戳**、**热词**、**声纹** 和 **多语言翻译** 等功能

<img src="./img/demo.gif" alt="Demo" width="400" height="200">

---

## 2. 安装

### Step 1: 下载模型
```bash
python download_models.py
```

### Step 2: 安装相关包并指定 ffmpeg 位置
请参考 `funsound.lib` 进行设置。

### Step 3: 修改配置
请参考 `funsound.config` 进行配置文件修改。

---

## 3. 启动服务

```bash
python -m funsound.web.server
```

---

## 4. 客户端测试

### Python 客户端
```bash
python client.py
```

### 网页客户端
修改 `client.html` 中的 WebSocket 服务地址进行测试。

---

## 5. 高级
Funsound交互逻辑如图：
<img src="./img/framework.png" alt="Funsound Framework" width="300" height="200">

- **funsound.brain.translator**: 基于大语言模型（LLM）的跨语言翻译
- **funsound.engine**: 并发引擎接口
- **funsound.pipeline**: Funsound 业务接口
- **funsound.web**: Funsound 服务部署

---

## 6. 联系作者

如有问题或建议，请联系：

**邮箱**: [605686962@qq.com](mailto:605686962@qq.com)

