#!/bin/bash
##############################################################################
# Step 0: 清理磁盘空间
# 运行方式: bash step0_cleanup.sh
# 
# 删除旧的uvcontsub和imaging产品，为新pipeline腾出空间
# 预计释放: ~100+ GB
##############################################################################

echo "================================================================"
echo "  Step 0: 清理磁盘空间"
echo "  $(date)"
echo "================================================================"
echo ""

MANUAL="/Volumes/HOU/NGC_3628/ALMA_data/2013.1.00087.S/science_goal.uid___A001_X144_X239/group.uid___A001_X144_X23a/member.uid___A001_X144_X23b/manual_calibration"
WORKDIR="$HOME/OneDrive/MasterThesis_HRL/work_dir/NGC3628"

echo "当前可用空间:"
df -h /Volumes/HOU | tail -1 | awk '{print "  "$4" 可用"}'
echo ""

# ---- 列出将要删除的文件 ----
echo "将要删除的文件:"
echo ""

echo "--- manual_calibration 目录中的旧contsub ---"
for f in "$MANUAL/NGC3628_concatenated.ms.contsub" \
         "$MANUAL/NGC3628_concatenated.ms.contsub_v2" \
         "$MANUAL/NGC3628_concatenated.ms.contsub_v3"; do
    if [ -d "$f" ]; then
        size=$(du -sh "$f" 2>/dev/null | awk '{print $1}')
        echo "  ❌ $f ($size)"
    fi
done

echo ""
echo "--- 工作目录中的旧imaging产品 ---"
for f in "$WORKDIR"/NGC3628_CO_H30alpha_combined* \
         "$WORKDIR"/NGC3628_CO_H30alpha_contsub* \
         "$WORKDIR"/IMAGING_WEIGHT_*; do
    if [ -e "$f" ]; then
        size=$(du -sh "$f" 2>/dev/null | awk '{print $1}')
        echo "  ❌ $(basename $f) ($size)"
    fi
done

echo ""
echo "⚠️  以下文件将被保留:"
echo "  ✅ $MANUAL/NGC3628_concatenated.ms (calibrated数据源)"
echo "  ✅ $MANUAL/uid___A002_X9630c0_X1719.ms.split.cal (EB1备份)"
echo "  ✅ $MANUAL/uid___A002_X9896b4_X1422.ms.split.cal (EB2备份)"
echo ""

read -p "确认删除以上文件？(yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "取消操作。"
    exit 0
fi

echo ""
echo "正在删除..."

# 删除旧的contsub
for f in "$MANUAL/NGC3628_concatenated.ms.contsub" \
         "$MANUAL/NGC3628_concatenated.ms.contsub_v2" \
         "$MANUAL/NGC3628_concatenated.ms.contsub_v3"; do
    if [ -d "$f" ]; then
        echo "  删除 $(basename $f)..."
        rm -rf "$f"
    fi
done

# 删除工作目录中的旧imaging产品
for pattern in "NGC3628_CO_H30alpha_combined" "NGC3628_CO_H30alpha_contsub" "IMAGING_WEIGHT_"; do
    for f in "$WORKDIR"/${pattern}*; do
        if [ -e "$f" ]; then
            echo "  删除 $(basename $f)..."
            rm -rf "$f"
        fi
    done
done

echo ""
echo "清理完成！"
echo "当前可用空间:"
df -h /Volumes/HOU | tail -1 | awk '{print "  "$4" 可用"}'
echo ""
echo "下一步: 运行 step1_uvcontsub.py 进行continuum subtraction"
