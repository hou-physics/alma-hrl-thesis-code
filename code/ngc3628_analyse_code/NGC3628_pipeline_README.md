# NGC 3628 H30α 完整Pipeline执行指南

## 总览

从已有的 `NGC3628_concatenated.ms` 重新做 uvcontsub + imaging + 分析。

**目标：** 复现Ada Table 4.2的结果 — H30α S/N=9.4, Flux=0.626 Jy·km/s

---

## 执行顺序

### Step 0: 清理磁盘空间 (~5分钟)

```bash
bash step0_cleanup.sh
```

删除旧的 contsub_v1/v2/v3 和旧imaging产品，释放 ~100+ GB。

**保留的文件：**
- `NGC3628_concatenated.ms` (48GB, 数据源)
- 两个单独EB的 `.ms.split.cal` (备份)

**清理后预期可用空间：** ~100 GB+

---

### Step 1: UV Plane Continuum Subtraction (~30分钟)

```bash
cd /Volumes/HOU/NGC_3628/ALMA_data/2013.1.00087.S/science_goal.uid___A001_X144_X239/group.uid___A001_X144_X23a/member.uid___A001_X144_X23b/manual_calibration

casa --nologger --nogui
# 在CASA中:
execfile('/path/to/step1_uvcontsub.py')
```

**关键改进（相比之前的版本）：**
1. ✅ 不使用 `combine='spw'` — 避免跨SPW外推导致斜坡
2. ✅ 用频率范围精确排除CO和H30α — 基于Ada的proc-commands.py
3. ✅ 排除SPW边缘 — 避免bandpass效应
4. ✅ fitorder=1 — 一阶多项式

**⚠️ 运行前必须做：**
- 检查 listobs 输出，确认 fitspw 中的频率范围与实际SPW范围匹配
- 如有偏差，修改脚本中的频率范围

**输出：** `NGC3628_concatenated.ms.contsub`

**质量检查：**
在CASA中用plotms检查contsub后的数据：
```python
plotms(vis='NGC3628_concatenated.ms.contsub',
       xaxis='frequency', yaxis='amplitude',
       avgtime='1e8', avgscan=True, field='NGC_3628')
```
- line-free区域的amplitude应该接近0
- CO线应该清晰可见
- H30α区域应该没有明显的斜坡或artifacts

---

### Step 2: Combined Cube Imaging (~1-3小时)

```python
# 继续在CASA中:
execfile('/path/to/step2_imaging.py')
```

**参数设置：**
- velocity范围: -1500 ~ +1500 km/s (300 channels × 10 km/s)
- rest frequency: 230.538 GHz (CO)
- CO大约在 ch209~254, H30α大约在 ch43~79
- Briggs weighting, robust=0.5
- 非交互模式 (niter=100000, threshold=1mJy)

**输出：** `NGC3628_CO_H30alpha_v4.image` + `.fits`

**质量检查（在CARTA中）：**
1. 打开 `NGC3628_CO_H30alpha_v4.fits`
2. 切到 CO channels (~ch220-240): 应该看到清晰的星系
3. 切到 H30α channels (~ch50-70): 应该看到微弱信号或纯噪声
4. 切到 line-free channels (~ch100-200): 应该是均匀噪声
5. ❌ 检查有无黑点 → 如有，需要调整tclean mask
6. ❌ 检查有无斜坡 → 如有，uvcontsub需要调整

**关键：** 在CARTA中找到一个 noise_box 区域（primary beam内、无信号），
记下坐标用于Step 3。

---

### Step 3: Ada方法分析 (~5分钟)

```bash
# 在普通Python环境中运行（不需要CASA）
python step3_analyze.py
```

**⚠️ 运行前必须做：**
1. 确认 `FITS_FILE` 路径正确
2. 确认 `CO_CHANS` 和 `H30A_CHANS` 范围（在CARTA中验证）
3. **更新 `NOISE_BOX` 坐标**（在CARTA中找到干净区域）
4. 确认 `SIGNAL_BOX` 包含星系

**输出：**
- `plots/NGC_3628_results_v4.txt` — 所有参数组合的结果
- `plots/NGC_3628_SN_heatmap_v4.png` — S/N热图

**结果判断：**
- S/N ≈ 7-12 → ✅ 合理（Ada得到9.4）
- S/N > 50 → ❌ 可能有问题（continuum未减干净、noise_box错误等）
- S/N < 0 → ⚠️ 非检测

---

## 常见问题排查

### 问题1: S/N异常大 (>50)
**原因：** continuum subtraction不充分
**解决：** 
- 检查uvcontsub的fitspw是否正确排除了谱线
- 检查noise_box是否在干净区域

### 问题2: 黑点/artifacts
**原因：** tclean的mask或clean参数问题
**解决：** 
- 调整tclean的mask范围
- 降低niter或提高threshold
- 尝试 `usemask='auto-multithresh'`

### 问题3: H30α区域有斜坡
**原因：** uvcontsub拟合不完美
**解决：** 
- 确认没有用 `combine='spw'`
- 检查fitspw的频率范围是否正确
- 尝试 fitorder=0（零阶，更保守）

### 问题4: 磁盘空间不足
**解决：** 
- 删除旧的contsub和imaging产品
- 如果两个EB的 `.ms.split.cal` 不再需要，可以删除节省空间
  （保留concatenated.ms即可）

---

## 关键参考

- **Ada论文 Section 3.6:** Map Smoothing方法
- **Ada论文 Section 4.3:** NGC 3628具体结果
- **Ada论文 Section 4.3.3:** Noise Peak调查
- **proc-commands.py:** Ada实际使用的CASA命令
- **David Lau Section 2.2.1:** 标准calibration+imaging流程
