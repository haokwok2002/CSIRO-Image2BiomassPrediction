import collections
import torch
import os
import json
from pathlib import Path

file_name = 'single_1209_ShadowAndGlare'

model_dir = Path('/data4/huangweigang/gh/csiro-biomass/历史模型/') / file_name
out_dir = Path(model_dir) / 'swa_models'
folds = [1, 2, 3, 4, 5]

for fold in folds:
    json_path = os.path.join(model_dir, f'model_top3_fold{fold}.json')
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found for fold {fold}: {json_path}")
        continue
        
    with open(json_path, 'r') as f:
        model_info = json.load(f)

    model_paths = set()
    epoch_set = set()
    
    for model_data in model_info.get('cv_top3', []):
        epoch = model_data['epoch']
        path = model_data['path']
        if epoch not in epoch_set:
            epoch_set.add(epoch)
            model_paths.add(path)

    for model_data in model_info.get('loss_top3', []):
        epoch = model_data['epoch']
        path = model_data['path']
        if epoch not in epoch_set:
            epoch_set.add(epoch)
            model_paths.add(path)
    
    model_paths = list(model_paths)
    
    print(f"\nFold {fold}: Found {len(model_paths)} unique models (after removing duplicate epochs)!")
    print(f"Unique epochs: {sorted(epoch_set)}")
    for path in model_paths:
        print(f" - {path}")

    if len(model_paths) < 2:
        print(f"⚠️  Fold {fold}: Only {len(model_paths)} unique models, skipping SWA...")
        continue

    models = []
    for module_path in model_paths:
        if os.path.exists(module_path):
            model = torch.load(module_path, map_location='cpu')
            models.append(model)
        else:
            print(f"❌ Model file not found: {module_path}")

    if len(models) < 2:
        print(f"⚠️  Fold {fold}: Only {len(models)} models loaded, skipping SWA...")
        continue

    worker_state_dicts = [m for m in models]
    weight_keys = list(worker_state_dicts[0].keys())
    print(f"Example weight keys: {list(weight_keys)[:5]}")

    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(models)):
            key_sum += worker_state_dicts[i][key]
        fed_state_dict[key] = key_sum / len(models)

    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'fold{fold}_swa.pt')
    torch.save(fed_state_dict, output_path)
    print(f"Fold {fold} averaging complete. Saved to: {output_path}")