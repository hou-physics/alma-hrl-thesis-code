##############################################################################
# Step 2: Combined Cube Imaging (CO + H30α)
#
# 运行方式: 在CASA 6.6中执行
#   cd /Volumes/HOU/NGC_3628/ALMA_data/2013.1.00087.S/.../manual_calibration
#   casa --nologger --nogui
#   execfile('/path/to/step2_imaging.py')
#
# 策略: 创建一个包含CO(2-1)和H30α的combined cube
#   - 避免grid mismatch问题
#   - CO mask可以直接应用到H30α（像素完全对齐）
#   - velocity范围: -1500 到 1500 km/s (以CO rest freq为参考)
#
# 参考: 
#   - 你之前的tclean参数 (上下文文档)
#   - Ada的proc-commands.py中的参数
#   - David Lau Section 2.2.1的流程
##############################################################################

import os
import time

# ============================================================================
# 配置
# ============================================================================

vis = 'NGC3628_concatenated.ms.contsub'
# tclean输出到本地SSD (避免外接硬盘的macOS ._文件问题)
local_workdir = os.path.expanduser('~/tclean_tmp/NGC3628')
final_workdir = '/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628'
workdir = final_workdir  # plots/logfiles仍存在外接盘

if not os.path.exists(local_workdir):
    os.makedirs(local_workdir)

imagename = os.path.join(local_workdir, 'NGC3628_CO_H30alpha_v5')

# 检查输入
if not os.path.exists(vis):
    print("ERROR: {} not found!".format(vis))
    print("请先运行 step1_uvcontsub.py")
    import sys; sys.exit(1)

# 创建工作目录
if not os.path.exists(workdir):
    os.makedirs(workdir)

print("="*70)
print("  Step 2: Combined Cube Imaging")
print("="*70)
print("")
print("输入: {}".format(vis))
print("输出: {}".format(imagename))
print("")

# ============================================================================
# 清理旧文件
# ============================================================================
for ext in ['.image', '.model', '.residual', '.pb', '.psf', '.sumwt', 
            '.mask', '.fits']:
    old = imagename + ext
    if os.path.exists(old):
        print("删除旧文件: {}".format(os.path.basename(old)))
        os.system('rm -rf ' + old)

# ============================================================================
# tclean 参数说明
# ============================================================================
#
# velocity范围设计:
#   NGC 3628 系统速度 ~830 km/s (z=0.002772)
#   CO rest freq = 230.538 GHz
#   H30α rest freq = 231.901 GHz
#   
#   以CO rest freq为参考:
#   H30α在CO参考系中的速度 ≈ c × (230.538 - 231.901×(1-0.002772)) / 230.538
#   ≈ c × (230.538 - 231.258) / 230.538
#   ≈ -937 km/s (相对于CO的位置)
#
#   所以cube需要覆盖:
#   - CO: 大约在 +700~+1000 km/s 附近 (系统速度区域)  
#   - H30α: 大约在 -900~-600 km/s 附近
#   
#   范围 -1500 到 +1500 km/s 足够覆盖两条线
#
# 成像参数:
#   - specmode='cube': 频谱模式
#   - width='10km/s': 每channel 10 km/s (与Ada一致)
#   - nchan=300: 覆盖 3000 km/s
#   - restfreq='230.538GHz': 以CO为参考频率
#   - outframe='LSRK': 标准参考系
#   - weighting='natural': 最大化灵敏度，恢复extended emission（与Ada/David一致）
#   - niter=10000, threshold='3mJy': Ada/David使用的参数
#   - deconvolver='hogbom': 标准点源deconvolver
#   - usemask='auto-multithresh': CASA自适应automask（效果接近interactive）
#   - pbcor=True: 直接输出pbcor cube
#
# ============================================================================

print("开始 tclean...")
print("这可能需要 1-3 小时，取决于机器性能。")
print("")

t_start = time.time()

tclean(
    vis         = vis,
    imagename   = imagename,
    field       = 'NGC_3628',
    spw         = '0,1,4,5',         # 包含CO和H30α的SPW
    specmode    = 'cube',
    start       = '-1500km/s',
    width       = '10km/s',
    nchan       = 300,
    restfreq    = '230.538GHz',       # CO(2-1) rest frequency
    outframe    = 'LSRK',
    veltype     = 'optical',
    imsize      = [512, 512],
    cell        = '0.2arcsec',
    weighting   = 'natural',           # 与Ada/David一致，最大化灵敏度
    deconvolver = 'hogbom',
    niter       = 10000,              # 与Ada/David一致 (原100000 over-cleaning)
    threshold   = '3mJy',            # 避免过深cleaning
    usemask     = 'auto-multithresh', # 自适应automask，替代interactive
    interactive = False,
    pbcor       = True,               # 直接输出pbcor cube
    restoringbeam = 'common'          # 统一beam，便于分析
)

t_end = time.time()
elapsed = (t_end - t_start) / 60.0

print("")
print("tclean 完成! 耗时: {:.1f} 分钟".format(elapsed))
print("")

# ============================================================================
# 导出FITS
# ============================================================================

if os.path.exists(imagename + '.image'):
    print("导出FITS文件...")
    # ⚠️ 重要: CASA 6.x中 tclean(pbcor=True) 的输出:
    #   .image       → 始终是 non-pbcor (原始 Jy/beam)
    #   .image.pbcor → 才是真正的 pbcor (PB校正后)
    #   .pb          → PB响应
    # 所以必须从 .image.pbcor 导出!

    # 导出 pbcor cube (主分析用)
    fitsname = imagename + '_pbcor.fits'
    exportfits(
        imagename = imagename + '.image.pbcor',
        fitsimage = fitsname,
        overwrite = True
    )
    print("FITS文件 (pbcor): {}".format(fitsname))

    # 同时导出 non-pbcor cube (用于mask生成)
    fitsname_nonpbcor = imagename + '_nonpbcor.fits'
    exportfits(
        imagename = imagename + '.image',
        fitsimage = fitsname_nonpbcor,
        overwrite = True
    )
    print("FITS文件 (non-pbcor): {}".format(fitsname_nonpbcor))

    # 同时导出PB响应 (用于生成non-pbcor mask)
    if os.path.exists(imagename + '.pb'):
        pb_fitsname = imagename + '_pb.fits'
        exportfits(
            imagename = imagename + '.pb',
            fitsimage = pb_fitsname,
            overwrite = True
        )
        print("PB文件: {}".format(pb_fitsname))
    else:
        print("⚠️ WARNING: .pb image未找到!")

    # 复制FITS到外接硬盘
    import shutil
    for src in [fitsname, fitsname_nonpbcor, pb_fitsname]:
        if os.path.exists(src):
            dst = os.path.join(final_workdir, os.path.basename(src))
            shutil.copy2(src, dst)
            print("已复制到: {}".format(dst))
    
    # 打印beam信息
    print("")
    print("--- Beam 信息 ---")
    header = imhead(imagename + '.image', mode='list')
    try:
        bmaj = header['beammajor']
        bmin = header['beamminor']
        bpa = header['beampa']
        print("  Beam major: {:.3f} {}".format(bmaj['value'], bmaj['unit']))
        print("  Beam minor: {:.3f} {}".format(bmin['value'], bmin['unit']))
        print("  Beam PA: {:.1f} {}".format(bpa['value'], bpa['unit']))
    except:
        print("  (多beam信息，使用common beam)")
        try:
            pbeams = header['perplanebeams']
            median_beam = pbeams.get('median area beam', pbeams.get('*0', {}))
            print("  Median/first beam: {}".format(median_beam))
        except:
            print("  无法获取beam信息，请用imhead手动检查")
    
    # 打印基本统计
    print("")
    print("--- Image 统计 ---")
    stats = imstat(imagename + '.image')
    print("  Max: {:.6f} Jy/beam".format(stats['max'][0]))
    print("  Min: {:.6f} Jy/beam".format(stats['min'][0]))
    print("  RMS: {:.6f} Jy/beam".format(stats['rms'][0]))
    
else:
    print("❌ ERROR: image未创建!")

print("")
print("="*70)
print("  Step 2 完成")
print("="*70)
print("")
print("下一步:")
print("  1. 在CARTA中打开 {} 检查cube质量".format(imagename + '.fits'))
print("  2. 检查以下内容:")
print("     - CO信号是否清晰 (大约在 channel 209~254)")
print("     - H30α区域是否干净 (大约在 channel 43~79)")
print("     - line-free channels的噪声是否均匀")
print("     - 有无黑点或条纹artifacts")
print("  3. 注意: 输出包含3个FITS文件:")
print("     - *_pbcor.fits:    pbcor cube (flux积分用)")
print("     - *_nonpbcor.fits: non-pbcor cube (mask生成用)")
print("     - *_pb.fits:       PB响应")
print("  4. 确认质量后运行 step3_analyze.py 进行分析")
