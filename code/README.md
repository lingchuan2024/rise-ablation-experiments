# RISE v14 消融实验 Python 包

此目录包含冻结 v14 工具实现、三个独立消融实验的代码与离线测试。实验设置、运行入口和服务器流程见仓库根目录的 [README](../README.md)。

在仓库根目录安装基础依赖：

```bash
python -m pip install -e code
```

需要本地神经重排器时，再安装相应依赖：

```bash
python -m pip install -r code/requirements-v10.txt
```

语料、检索索引和模型权重由运行环境单独提供，不包含在 Python 包中。各运行命令通过参数指定资源路径；认证信息通过进程环境传入。

上游署名与许可原文见 [LICENSE](LICENSE) 及根目录的 [NOTICE](../NOTICE.md)。
