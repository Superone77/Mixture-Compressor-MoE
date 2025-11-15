# create_bit_config_2.5.py
import pickle

# Mixtral-8x7B 通常有 32 层，根据你的模型调整
num_layers = 32  # 如果不同，请修改
num_experts = 8

# 为了达到平均比特 2.5（当 wbits=3 时）：
# - 4 个专家用 2 比特 (wbits-1, 标记为 1)
# - 4 个专家用 3 比特 (wbits, 标记为 2 或其他值)
# 平均 = (4*2 + 4*3) / 8 = 2.5

bit_config = {}
for layer_idx in range(num_layers):
    bit_config[layer_idx] = {}
    # 前 4 个专家使用低比特 (标记为 1，实际使用 wbits-1)
    for expert_idx in range(4):
        bit_config[layer_idx][expert_idx] = 1
    # 后 4 个专家使用中比特 (标记为 2，实际使用 wbits)
    for expert_idx in range(4, 8):
        bit_config[layer_idx][expert_idx] = 2

# 保存为 pickle 文件
output_path = 'bit_config_2.5.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(bit_config, f)

print(f"✓ Created bit_config file: {output_path}")
print(f"  - {num_layers} layers")
print(f"  - 4 experts per layer use low bit (wbits-1)")
print(f"  - 4 experts per layer use medium bit (wbits)")
print(f"  - Average bit: 2.5 (when wbits=3)")