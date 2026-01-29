import os

# 你要测试的 freeze_ratio 值
freeze_values = [0.45, 0.50, 0.55, 0.60, 0.65]

# 原始文件名（无需写路径，自动查找）
src_file = "/data4/huangweigang/gh/csiro-biomass/1114/freeze_ratio_test/DM_pipeline_1114原版fix.py"

# 自动找到原始文件的绝对路径
src_path = os.path.abspath(src_file)

# 自动获取原始文件所在目录
src_dir = os.path.dirname(src_path)

print(f"原文件位置：{src_path}")
print(f"生成文件将放在：{src_dir}")

# 读取原始文件内容
with open(src_path, "r", encoding="utf-8") as f:
    original_code = f.read()

for fr in freeze_values:
    # 生成新文件名
    dst_file = f"DM_pipeline_1114原版fix_fr{int(fr*100):03d}.py"

    # 拼接到原目录下
    dst_path = os.path.join(src_dir, dst_file)

    # 替换 freeze_ratio 那一行的示例
    new_code = original_code.replace(
        '"freeze_ratio"                   : 0.6',
        f'"freeze_ratio"                   : {fr}'
    )

    # 写入新文件
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(new_code)

    print(f"已生成：{dst_path}")

print("全部脚本生成完毕！")
