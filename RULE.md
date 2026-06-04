# 工作规则

## Commit
- 不要 Co-Authored-By，个人工作不需要署名

## 代码
- 能简则简，不要过度设计目录结构（文件不多就扁平放）
- 经常变的配置（输出路径、日志路径）由代码自动处理，不要写在 shell 里

## WORK_LOG
- 按日期分块
- Experiments / Changes / Fixes / Decisions / Cleanup 分段
- 核心改动和常规整理分开

## 实验
- OPOPE prompt: "Describe this image in detail."（简洁中性）
- max_new_tokens: 512（保证描述完整）
- GPU 0,1,2 可用
