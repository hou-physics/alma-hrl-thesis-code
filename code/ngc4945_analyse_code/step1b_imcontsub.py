##############################################################################
# Step 1b: Image-Domain Continuum Subtraction for NGC 4945
#
# 对已有的 product cube 做 imcontsub，不需要 calibrated MS。
# 适用场景: archive product 已有 cube，想快速测试 HRL 检测。
#
# 运行方式: 在 CASA 6.x 中执行
#   casa --nologger --nogui
#   execfile('step1b_imcontsub_NGC4945.py')
#
# 输入: NGC4945_H30a_spw1_v1.fits (reverse-pbcor cube, 无 contsub)
# 输出: NGC4945_H30a_spw1_v1_contsub.fits (扣除连续谱后的 cube)
#
# 原理:
#   imcontsub 对每个像素沿频率轴做多项式拟合 (用 line-free channels)，
#   然后从原始 cube 减去拟合的连续谱模型。
#   优点: 不需要 MS，直接对 image 操作
#   缺点: 不如 uvcontsub 准确 (在 image domain 拟合而非 UV plane)
##############################################################################

import os

# ============================================================================
# 配置
# ============================================================================

workdir = '/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945'

fitsfile = os.path.join(workdir, 'NGC4945_H30a_spw1_v1.fits')
casaimage = os.path.join(workdir, 'NGC4945_H30a_spw1_v1.im')
contsubimage = os.path.join(workdir, 'NGC4945_H30a_spw1_v1_contsub.im')
contmodel = os.path.join(workdir, 'NGC4945_H30a_spw1_v1_cont.im')
outfits = os.path.join(workdir, 'NGC4945_H30a_spw1_v1_contsub.fits')

# ============================================================================
# Cube 参数 (来自 FITS header)
# ============================================================================
#
# 240 channels, 230.827 - 232.695 GHz, ~7.8 MHz/ch (~10 km/s)
#
# H30alpha:
#   Rest freq: 231.9009 GHz
#   NGC 4945 z = 0.00188
#   Observed freq: 231.4657 GHz
#   -> Channel ~83
#
# Line-free channels 选择策略:
#   避开 H30a 附近 (channel 60-110, 保守范围)
#   也要避开其他可能的线 (README 说 "multiple spectral lines detected")
#
#   安全的 line-free 区间:
#     Channel 0-50:   230.83 - 231.22 GHz (cube 左侧)
#     Channel 120-240: 231.76 - 232.69 GHz (cube 右侧)
#
#   注意: README 提到这个 SPW 可能有多条谱线。
#   如果 imcontsub 结果不好，可能需要进一步缩小 line-free 范围。
# ============================================================================

# line-free channels (避开 H30a 和可能的其他线)
linefree = '0~50;120~239'

print("="*70)
print("  Step 1b: Image-Domain Continuum Subtraction")
print("  Galaxy: NGC 4945")
print("  Target: H30alpha @ 231.466 GHz (channel ~83)")
print("="*70)
print("")

# ============================================================================
# Step 1: 导入 FITS -> CASA image
# ============================================================================

if os.path.exists(casaimage):
    print("删除旧的 CASA image: {}".format(os.path.basename(casaimage)))
    os.system('rm -rf ' + casaimage)

print("导入 FITS -> CASA image...")
importfits(
    fitsimage = fitsfile,
    imagename = casaimage,
    overwrite = True
)
print("  -> {}".format(casaimage))

# 验证导入
imhead(casaimage, mode='summary')
print("")

# ============================================================================
# Step 2: imcontsub
# ============================================================================

# 清理旧输出
for f in [contsubimage, contmodel]:
    if os.path.exists(f):
        print("删除旧文件: {}".format(os.path.basename(f)))
        os.system('rm -rf ' + f)

print("运行 imcontsub...")
print("  Line-free channels: {}".format(linefree))
print("  Fit order: 1 (线性拟合)")
print("  这可能需要几分钟...")
print("")

imcontsub(
    imagename = casaimage,
    linefile  = contsubimage,
    contfile  = contmodel,
    fitorder  = 1,
    chans     = linefree
)

print("")
if os.path.exists(contsubimage):
    print("imcontsub 完成!")

    # 基本统计
    stats = imstat(contsubimage)
    print("  Contsub cube 统计:")
    print("    Max: {:.6f} Jy/beam".format(stats['max'][0]))
    print("    Min: {:.6f} Jy/beam".format(stats['min'][0]))
    print("    RMS: {:.6f} Jy/beam".format(stats['rms'][0]))
else:
    print("ERROR: imcontsub 输出未创建!")
    import sys; sys.exit(1)

# ============================================================================
# Step 3: 导出为 FITS
# ============================================================================

print("")
print("导出 contsub cube -> FITS...")
exportfits(
    imagename = contsubimage,
    fitsimage = outfits,
    overwrite = True
)
print("  -> {}".format(outfits))

# ============================================================================
# Step 4: 快速质量检查 — 提取核心光谱
# ============================================================================

print("")
print("提取核心区域光谱做快速检查...")

# NGC 4945 核心大约在图像中心
# cube 是 660x660, 中心 ~330,330
# 取一个小 box
box_center = '325,325,335,335'  # 10x10 pixel box

stats_spec = imstat(contsubimage, box=box_center, axes=[0,1])
if 'mean' in stats_spec:
    spec = stats_spec['mean']
    print("  核心区域平均光谱 (前10个和H30a附近的channels):")
    for i in range(min(10, len(spec))):
        print("    ch {:3d}: {:.6f} Jy/beam".format(i, spec[i]))
    print("    ...")
    for i in range(max(0, 75), min(95, len(spec))):
        print("    ch {:3d}: {:.6f} Jy/beam".format(i, spec[i]))

print("")
print("="*70)
print("  Step 1b 完成!")
print("="*70)
print("")
print("输出文件:")
print("  Contsub cube: {}".format(outfits))
print("  Cont model:   {}".format(contmodel))
print("")
print("下一步:")
print("  1. 在 CARTA 中打开 {} 检查质量".format(os.path.basename(outfits)))
print("     - line-free channels 应该接近 0")
print("     - H30a 附近 (channel ~83) 应该能看到信号")
print("  2. 如果质量OK，运行 step3 分析脚本")
print("  3. 如果有明显残余或artifacts，尝试调整 linefree 范围或 fitorder")
