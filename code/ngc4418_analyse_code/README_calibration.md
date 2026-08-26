# NGC 4418 ASDM scenario C 工作流

Project 2012.1.00377.S MOUSID `uid://A002/X5d7935/X14b` 的可用数据只有 raw
ASDM + 校准/辅助 tars——没有现成 calibrated MS，更没有 imaged FITS。需要
完整跑 ALMA 校准 + imaging 流水线，再进 step3。

## 0. 下载 (15-16 GB)

```bash
cd /Volumes/HouAstro/master/master_thesis/my_code/ngc4418_analyse_code
conda run -n casa_env --no-capture-output python step-1_download.py
```

下载后 `work_dir/NGC4418/` 应有：
- `2012.1.00377.S_uid___A002_Xa5df2c_X48ee.asdm.sdm.tar` (~15.4 GB) — raw 可视性
- `2012.1.00377.S_uid___A002_X5d7935_X14b_001_of_001.tar` (~161 MB) — calibration scripts + 参考数据
- `2012.1.00377.S_uid___A002_X5d7935_X14b_auxiliary.tar` (~71 MB) — 辅助 (logs, weblog)

## 1. 解压

```bash
cd /Volumes/HouAstro/master/master_thesis/work_dir/NGC4418
for f in *.tar; do tar xf "$f"; done
```

解压后会有 ALMA 标准目录树：
```
2012.1.00377.S/
└── science_goal.uid___...../group.uid___...../member.uid___A002_X5d7935_X14b/
    ├── raw/                  # ASDM 原数据
    ├── calibration/          # scriptForPI.py / scriptForCalibration.py
    ├── log/
    └── README
```

## 2. 跑 calibration（CASA, 1-3 h）

ALMA 项目自带 `scriptForPI.py`（统一入口）或 `scriptForCalibration.py`（按 EB 分别校准）。
推荐用 scriptForPI——它内部按顺序调 scriptForCalibration、做 fluxscale、做 split。

```bash
cd /Volumes/HouAstro/master/master_thesis/work_dir/NGC4418/.../calibration

# 启动 CASA（pipeline 模式）
/path/to/casa --nologger --nogui --pipeline

# 在 CASA 中：
execfile('scriptForPI.py')
```

### 输出

`calibrated_final.ms`（或 `*.ms.split.cal`），**一般 5-15 GB**——bandpass / 相位 /
振幅校准都做完了，line-free 通道扣 continuum 之前的状态。

### 若 scriptForPI 报错

- "找不到 reference data" → 确认 calibration tar 解压完整（含 `*.flagcal` / `*.bpcal` 等参考）
- "ASDM 未找到" → 检查解压目录里 `raw/` 是否含 `*.asdm.sdm`
- "CASA 版本不兼容" → 该项目 2012-cycle，需要 CASA 4.x 或 5.0；按 README 注明的版本切换

## 3. UV-plane continuum subtraction（CASA, 30 min）

参考 `ngc3628_analyse_code/step1_uvcontsub.py` 的模板，关键点：

```python
# 在 CASA 中：
# 1. 用 listobs 看 SPW 列表，确定 H30α 和 CO(2-1) 在哪个 SPW
listobs(vis='calibrated_final.ms')

# 2. 用频率范围排除两条线（NOT spw 通道 — 跨 SPW 外推会出斜坡）
#    H30α 观测频率 = 231.901 / (1+0.007085) = 230.269 GHz
#    CO(2-1)  观测频率 = 230.538 / (1+0.007085) = 228.916 GHz
#    各排 ±300 km/s ≈ ±230 MHz
fitspw_h30a = '<spw_id>:230.039~230.499GHz'  # 排除窗口
fitspw_co21 = '<spw_id>:228.686~229.146GHz'

uvcontsub(
    vis='calibrated_final.ms',
    fitspw=f'{fitspw_h30a};{fitspw_co21}',
    fitorder=1,             # 一阶多项式
    excludechans=False,
    combine='',             # 关键：不要 combine='spw'
    want_cont=False,
)
# 输出: calibrated_final.ms.contsub
```

## 4. tclean imaging（CASA, 1-3 h）

参考 `ngc3628_analyse_code/step2_imaging.py`。NGC 4418 推荐参数：

```python
import numpy as np

REST_GHZ = {'H30a': 231.900928, 'CO21': 230.538}
Z = 0.007085

for line, rest in REST_GHZ.items():
    obs_GHz = rest / (1 + Z)
    tclean(
        vis='calibrated_final.ms.contsub',
        imagename=f'NGC4418_{line}_v1',
        cell='0.05arcsec',         # NGC 4418 sub-arcsec, beam ~0.23"
        imsize=512,                # 25.6" field (NGC 4418 紧凑核 < 1")
        specmode='cube',
        outframe='LSRK',
        restfreq=f'{rest}GHz',
        start='-500km/s',          # ±500 km/s 包线
        width='5km/s',             # 通道宽度
        nchan=200,
        weighting='briggs',
        robust=0.5,
        niter=100000,
        threshold='0.5mJy',
        usemask='auto-multithresh',
        deconvolver='multiscale',
        scales=[0, 5, 15],
        pblimit=0.2,
        pbcor=True,                # 出 pbcor 版
    )
    # 同时出 nonpbcor 版给 mask 用
    # exportfits(imagename=f'NGC4418_{line}_v1.image', fitsimage=...)
```

## 5. 标准化 symlinks + 进 step3

下游 `_step3` 包要求 `work_dir/NGC4418/NGC4418_{line}_{pbcor|nonpbcor|pb}.fits`
命名约定。tclean 出来后 ln -s 一下：

```bash
cd /Volumes/HouAstro/master/master_thesis/work_dir/NGC4418
ln -s NGC4418_H30a_v1.image.pbcor.fits     NGC4418_H30a_pbcor.fits
ln -s NGC4418_H30a_v1.image.fits           NGC4418_H30a_nonpbcor.fits
ln -s NGC4418_H30a_v1.pb.fits.gz           NGC4418_H30a_pb.fits.gz
ln -s NGC4418_CO21_v1.image.pbcor.fits     NGC4418_CO21_pbcor.fits
ln -s NGC4418_CO21_v1.image.fits           NGC4418_CO21_nonpbcor.fits
ln -s NGC4418_CO21_v1.pb.fits.gz           NGC4418_CO21_pb.fits.gz
```

然后 `/analyze-galaxy NGC4418 line=H30a z=0.007085`。

## 时间预算

| 阶段 | 预计 | 备注 |
|---|---|---|
| download | 30 min - 2 h | 取决于网速 |
| 解压 | 5-10 min | |
| scriptForPI | 1-3 h | CASA 单线程 |
| uvcontsub | 20-40 min | |
| tclean H30α + CO(2-1) | 1-3 h | 总和 |
| step3 分析 | 5-10 min | 自动 |
| **合计** | **3-9 h** | "一天分析一个星系"预算合理 |
