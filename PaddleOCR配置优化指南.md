# PaddleOCR配置优化指南

## 📋 官方参数说明

基于PaddleOCR官方文档，以下是关键参数的详细说明和最佳实践：

### 🎯 检测参数

#### 1. det_db_thresh (像素分类阈值)
- **作用**: DB输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点
- **默认值**: 0.3
- **优化值**: 0.2
- **原理**: 降低阈值可以检测更多文本区域，对小文字和模糊文字效果更好

#### 2. det_db_box_thresh (文本框置信度阈值)  
- **作用**: 检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域
- **默认值**: 0.6
- **优化值**: 0.4
- **原理**: 降低阈值减少漏检，特别适合建筑图纸中的小字体

#### 3. det_db_unclip_ratio (文本框扩张系数)
- **作用**: 使用Vatti clipping算法对文字区域进行扩张
- **默认值**: 1.5
- **优化值**: 2.5
- **原理**: 官方推荐值，避免文字边缘被裁剪，确保完整文字识别

#### 4. use_dilation (膨胀处理)
- **作用**: 对分割结果进行膨胀以获取更优检测效果
- **默认值**: False
- **优化值**: True
- **原理**: 官方推荐开启，能有效改善检测效果

#### 5. det_db_score_mode (得分计算模式)
- **作用**: DB的检测结果得分计算方法
- **选项**: 'fast' / 'slow'
- **默认值**: 'fast'
- **优化值**: 'slow'
- **原理**: 'slow'模式更精确，适合弯曲文字和复杂场景

### 🔤 识别参数

#### 6. drop_score (识别置信度过滤)
- **作用**: 识别结果置信度阈值，低于此值的结果会被过滤
- **默认值**: 0.5
- **优化值**: 0.3
- **原理**: 适当降低阈值保留更多有效识别结果

#### 7. use_angle_cls (角度分类器)
- **作用**: 是否启用文字角度分类
- **默认值**: False
- **优化值**: True
- **原理**: 处理旋转文字，提高识别准确率

## 🏗️ 建筑图纸专用配置

```python
ocr = PaddleOCR(
    lang='ch',                    # 中文识别
    det_db_thresh=0.2,           # 更敏感的像素检测
    det_db_box_thresh=0.4,       # 减少漏检
    det_db_unclip_ratio=2.5,     # 官方推荐扩张系数
    drop_score=0.3,              # 识别结果置信度过滤
    use_angle_cls=True,          # 启用角度分类器
    cls_thresh=0.8,              # 角度分类阈值
    use_dilation=True,           # 膨胀处理提升效果
    det_db_score_mode='slow'     # 精确得分计算
)
```

## 📊 参数影响对比

| 参数 | 默认值 | 优化值 | 效果 |
|------|--------|--------|------|
| det_db_thresh | 0.3 | 0.2 | 检测更多小文字 |
| det_db_box_thresh | 0.6 | 0.4 | 减少漏检 |
| det_db_unclip_ratio | 1.5 | 2.5 | 避免边缘丢失 |
| use_dilation | False | True | 改善检测效果 |
| det_db_score_mode | 'fast' | 'slow' | 更精确计算 |

## 🔧 针对不同问题的调优建议

### 问题1: 小文字检测不到
- 降低 `det_db_thresh` 到 0.1-0.2
- 降低 `det_db_box_thresh` 到 0.3-0.4
- 启用 `use_dilation=True`

### 问题2: 文字边缘被裁剪
- 增大 `det_db_unclip_ratio` 到 2.5-3.0
- 使用 `det_db_score_mode='slow'`

### 问题3: 识别结果太多噪声
- 提高 `drop_score` 到 0.4-0.5
- 适当提高 `det_db_box_thresh`

### 问题4: 旋转文字识别差
- 启用 `use_angle_cls=True`
- 降低 `cls_thresh` 到 0.7-0.8

## 📚 官方文档参考

- [PaddleOCR参数文档](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [DB算法参数说明](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/blog/inference_args.md)
- [最佳实践FAQ](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/FAQ.md)

## 🎯 性能优化建议

1. **CPU优化**: 使用 `use_gpu=False` 确保CPU运行稳定
2. **内存管理**: 大图片可以分块处理避免内存溢出
3. **预处理**: 图像增强可以提高识别效果
4. **后处理**: 结合业务逻辑过滤无效结果

## 🧪 测试验证

当前配置在以下测试案例中表现：
- ✅ demo.jpg: 成功识别"厨房"
- ✅ demo2.jpg: 识别多个房间名称  
- ✅ demo3.jpg: 优化后成功识别"厨房"（之前失败）
- ❌ demo4.jpg: 图像质量较差，需要进一步优化

建议针对具体项目进行参数微调，以达到最佳效果。
