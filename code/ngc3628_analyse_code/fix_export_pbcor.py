##############################################################################
# Fix: 从本地SSD的tclean输出重新导出真正的pbcor cube
#
# 运行方式: 在CASA 6.x中运行
#   execfile('fix_export_pbcor.py')  或  exec(open('fix_export_pbcor.py').read())
#
# 背景: tclean(pbcor=True) 会生成:
#   .image       → 非pbcor (原始Jy/beam)
#   .image.pbcor → pbcor   (PB校正后的Jy/beam)
#   .pb          → PB响应
#
# 之前step2错误地从.image导出，得到的是non-pbcor cube。
# 此脚本从.image.pbcor导出，得到真正的pbcor cube。
##############################################################################

import os
import shutil

# === 路径配置 ===
SSD_DIR = os.path.expanduser('~/tclean_tmp/NGC3628')
DEST_DIR = '/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628'

imagename = os.path.join(SSD_DIR, 'NGC3628_CO_H30alpha_v5')

# 检查文件是否存在
pbcor_image = imagename + '.image.pbcor'
nonpbcor_image = imagename + '.image'
pb_image = imagename + '.pb'

for f in [pbcor_image, nonpbcor_image, pb_image]:
    if os.path.exists(f):
        print("OK: {}".format(f))
    else:
        print("MISSING: {} — 请检查路径!".format(f))

# === 导出pbcor cube ===
pbcor_fits = os.path.join(DEST_DIR, 'NGC3628_CO_H30alpha_v5_pbcor.fits')
print("\n--- 导出pbcor cube ---")
print("  from: {}".format(pbcor_image))
print("  to:   {}".format(pbcor_fits))
exportfits(
    imagename = pbcor_image,
    fitsimage = pbcor_fits,
    overwrite = True
)
print("  done!")

# === 同时重命名原来的v5.fits为non-pbcor，避免混淆 ===
old_v5 = os.path.join(DEST_DIR, 'NGC3628_CO_H30alpha_v5.fits')
new_nonpbcor = os.path.join(DEST_DIR, 'NGC3628_CO_H30alpha_v5_nonpbcor.fits')
if os.path.exists(old_v5):
    print("\n--- 重命名原v5.fits → v5_nonpbcor.fits ---")
    os.rename(old_v5, new_nonpbcor)
    print("  done!")

# === 验证 ===
print("\n--- 验证 ---")
imstat_pbcor = imstat(imagename=pbcor_image)
imstat_nonpbcor = imstat(imagename=nonpbcor_image)
print("pbcor    max={:.6f}, min={:.6f}, rms={:.6f}".format(
    imstat_pbcor['max'][0], imstat_pbcor['min'][0], imstat_pbcor['rms'][0]))
print("nonpbcor max={:.6f}, min={:.6f}, rms={:.6f}".format(
    imstat_nonpbcor['max'][0], imstat_nonpbcor['min'][0], imstat_nonpbcor['rms'][0]))
print("\n如果pbcor的rms明显大于nonpbcor的rms，说明导出正确（边缘噪声被PB放大）")

print("\n=== 完成! ===")
print("新文件: {}".format(pbcor_fits))
print("原文件已重命名: {}".format(new_nonpbcor))
