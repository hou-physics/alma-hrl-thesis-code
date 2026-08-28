##############################################################################
# Step 1: UV Plane Continuum Subtraction (CASA 6.6 版本)
#
# 运行方式: 在CASA 6.6中执行
#   cd 到manual_calibration目录
#   casa --nologger --nogui
#   execfile('step1_uvcontsub.py')
#
# ⚠️ CASA 6.6 的 uvcontsub 是新版本:
#   - 参数名: fitspec (不是 fitspw)
#   - 需要指定: outputvis (不自动生成 .contsub)
#   - 不支持: combine, want_cont, solint
#   - 如需旧版功能，使用 uvcontsub_old
##############################################################################

import os

vis = 'NGC3628_concatenated.ms'
outputvis = 'NGC3628_concatenated.ms.contsub'

if not os.path.exists(vis):
    print("ERROR: {} not found!".format(vis))
    import sys; sys.exit(1)

print("="*70)
print("  Step 1: UV Plane Continuum Subtraction (CASA 6.6)")
print("="*70)
print("")

# ---- listobs ----
print("--- 运行 listobs ---")
listobs(vis=vis, listfile='NGC3628_listobs_check.txt', overwrite=True)
print("已保存到 NGC3628_listobs_check.txt")
print("")

# ---- 清理旧文件 ----
if os.path.exists(outputvis):
    print("删除旧的 {}...".format(outputvis))
    os.system('rm -rf ' + outputvis)

# ============================================================================
# fitspec 参数
#
# CASA 6.6 新版 uvcontsub 用 fitspec 代替 fitspw
# 语法相同: 'spw:频率范围' 或 'spw:channel范围'
#
# SPW配置:
#   SPW 0/4: ~228.44-230.30 GHz, 包含CO(2-1)在~229.90 GHz
#   SPW 1/5: ~230.81-232.69 GHz, 包含H30α在~231.26 GHz
#
# 策略: 指定line-free channels用于拟合continuum
#   SPW 0: 228.49~229.55 GHz (CO左侧，排除SPW边缘和CO)
#   SPW 1: 230.86~231.00 GHz; 231.55~232.64 GHz (H30α两侧)
#   SPW 4: 同SPW 0
#   SPW 5: 同SPW 1
#
# ⚠️ 运行前请先检查 NGC3628_listobs_check.txt
#    确认SPW的实际频率范围，如有偏差请调整下面的值！
# ============================================================================

fitspec_str = (
    '0:228.49~229.55GHz,'
    '1:230.86~231.00GHz;231.55~232.64GHz,'
    '4:228.49~229.55GHz,'
    '5:230.86~231.00GHz;231.55~232.64GHz'
)

print("fitspec = '{}'".format(fitspec_str))
print("fitorder = 1")
print("outputvis = '{}'".format(outputvis))
print("")
print("开始uvcontsub... (预计20-40分钟)")
print("")

uvcontsub(
    vis       = vis,
    outputvis = outputvis,
    field     = 'NGC_3628',
    spw       = '0,1,4,5',
    fitspec   = fitspec_str,
    fitorder  = 1
)

print("")
if os.path.exists(outputvis):
    print("✅ uvcontsub 完成! 输出: {}".format(outputvis))
    listobs(vis=outputvis, listfile='NGC3628_contsub_listobs.txt', overwrite=True)
    print("listobs 已保存")
else:
    print("❌ ERROR: 输出文件未创建!")

print("")
print("="*70)
print("下一步:")
print("  1. 用plotms检查质量 (在CASA GUI中):")
print("     plotms(vis='{}',".format(outputvis))
print("            xaxis='frequency', yaxis='amplitude',")
print("            avgtime='1e8', avgscan=True, field='NGC_3628')")
print("  2. 确认line-free区域接近0，CO清晰，H30α区域干净")
print("  3. 运行 step2_imaging.py")
print("="*70)