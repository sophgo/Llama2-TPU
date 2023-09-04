![](./assets/sophgo_chip.png)

# Llama2-TPU

本项目实现BM1684X部署语言大模型[Llama2-6B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

下文中默认是PCIE环境；如果是SoC环境，按提示操作即可。


## 开发环境


1. 下载docker，启动容器，如下：

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
后文假定环境都在docker的`/workspace`目录。

如果是要在SoC环境运行，则需要安装如下库：

``` shell
apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

2. 下载`Llama2-7B`，比较大，会花较长时间

下载路径为: http://disk-sophgo-vip.quickconnect.to/sharing/RAcn5E1zU

3. 修改源代码，修改的目的是为了保证model\_tool --combine的时候block和block\_cache权重能对齐

```shell
pip show transformers
```

找到transformers库的位置

```shell
vi /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```

修改316行左右的代码，修改前为

```python
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
```

修改后：

```python
if past_key_value is not None:
  cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len-1)
else:
  cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
```

3. 下载`TPU-MLIR`代码并编译，(也可以直接下载编译好的release包解压)

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

4. 下载[sentencepiece](https://github.com/google/sentencepiece)，并编译得到`sentencepiece.a`

```shell
git clone git@github.com:google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j
```

如果要编译SoC环境，则需要在`CMakeLists.txt`加入如下代码：

```cmake
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
```

5. 下载libsophon库并安装

如果是跑分布式模型，libsophon请联系yi.chu@sophgo.com获取

在算能官网<https://developer.sophgo.com/site/index/material/all/all.html>可以找到SDK最新版本，如下：

```shell
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/06/15/16/Release_230501-public.zip
```
解压sdk后安装libsophon，如下：

```shell
apt install sophon-libsophon-dev_0.4.8_amd64.deb
```

注意如果是SoC环境则安装arm64版本`sophon-libsophon-dev_0.4.8_arm64.deb`

6. 下载本项目`Llama2-TPU`，如下：

``` shell
git clone git@github.com:sophgo/Llama2-TPU.git
```

## 编译模型(分布式)

1. 导出所有onnx模型，如果过程中提示缺少某些组件，直接`pip install 组件`即可

``` shell
cd Llama2-TPU/compile
python3 export_onnx.py
```
此时有大量onnx模型被导出到tmp目录

2. 对onnx模型进行编译，生成bmodel，这个过程会花一些时间，最终生成`llama2-7b.bmodel`文件

```shell
./compile.sh --num_device 2
```

## 编译程序(C++版本)

```shell
cd Llama2-TPU/demo
mkdir build
cd build
cmake ..
make -j
```

如果是SoC环境，则将CMakeLists.txt中加入，并将SoC版本的`libsentencepiece.a`替换过来：

```cmake
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
```

编译生成llama2可执行程序，将`llama2`、`llama2-7b.bmodel`和`tokenizer.model`拷贝到运行环境就可以执行了。
(`tokenizer.model`来自`https://huggingface.co/meta-llama/Llama-2-7b-chat-hf`)

运行指令
```shell
./llama2 --dev_id 0,1
```

