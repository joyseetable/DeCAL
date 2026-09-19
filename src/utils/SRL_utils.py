# 文件路径: src/utils/srl_parser.py

import json
from transformers import pipeline
from tqdm import tqdm
import os
import torch

class SRLParser:
    def __init__(self, model_name="liaad/srl-en_mbert-base"): # <-- 使用你亲自找到并验证过的模型！
        print(f"--- 正在加载 SRL 模型: {model_name} (这可能需要一些时间)... ---")
        
        # 确定设备
        device = 0 if torch.cuda.is_available() else -1
        
        # 使用 "token-classification" 任务加载 SRL 模型
        self.srl_pipeline = pipeline(
            "token-classification",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple", # "simple" 策略会自动拼接 B-ARG0 和 I-ARG0
            device=device
        )
        print("✅ SRL 模型加载完成。")

    def parse_sentence(self, sentence: str):
        """
        解析单个句子，提取 实体(ARG0)、关系(V)、场景(ARG1/ARGM-LOC)。
        """
        try:
            srl_results = self.srl_pipeline(sentence)
            if not srl_results:
                return {'entity': None, 'relation': None, 'scene': None}
        except Exception as e:
            # print(f"Warning: SRL analysis failed for '{sentence}'. Error: {e}")
            return {'entity': None, 'relation': None, 'scene': None}

        parsed = {'entity': None, 'relation': None, 'scene': None}
        
        for item in srl_results:
            group = item['entity_group']
            word = item['word'].strip()
            
            if not word: continue

            # 只取第一个匹配到的角色，以保持简单和稳定
            if 'ARG0' in group and parsed['entity'] is None:
                parsed['entity'] = word
            elif 'V' in group and parsed['relation'] is None:
                parsed['relation'] = word
            elif 'ARG1' in group and parsed['scene'] is None:
                parsed['scene'] = word
            elif 'ARGM-LOC' in group and parsed['scene'] is None: # 作为 ARG1 的备胎
                parsed['scene'] = word
                
        return parsed

    def process_flickr30k_annotations(self, input_json_path: str, output_json_path: str):
        """
        读取原始的 Flickr30k JSON 文件，进行 SRL 解析，并保存为新的 JSON 文件。
        """
        if os.path.exists(output_json_path):
            print(f"✅ SRL 标注文件已存在于: {output_json_path}。跳过处理。")
            return

        with open(input_json_path, 'r') as f:
            data = json.load(f)['images']

        sentence_count = sum(len(img.get('sentences', [])) for img in data)
        print(f"--- 开始对 {len(data)} 张图片, 共 {sentence_count} 条描述进行 SRL 解析... ---")
        
        # 逐个图片处理，以保证健壮性
        for image_data in tqdm(data, desc="Processing Images"):
            if 'sentences' in image_data:
                for sentence_data in image_data['sentences']:
                    # 调用我们健壮的单句解析函数
                    parsed_roles = self.parse_sentence(sentence_data['raw'])
                    sentence_data['srl'] = parsed_roles
        
        output_data = {"images": data}
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"--- ✅ SRL 解析完成。结果已保存到: {output_json_path} ---")


if __name__ == '__main__':
    # --- 最后的、决定性的步骤 ---
    # 1. 请在这里，填上你服务器上 dataset_flickr30k.json 的真实、绝对路径
    original_annotations = "/data/clc/APSE-IPIK/NEW_APSEIPIK/DATA/flickr30k/annotations/train.json"
    
    # 2. 请在这里，填上你想要保存新文件的真实、绝对路径
    srl_annotations = "/data/clc/APSE-IPIK/NEW_APSEIPIK/DATA/flickr30k/annotations/annotations_srl.json"
    
    if "/path/to/your/" in original_annotations:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 请在运行前，修改脚本中的 original_annotations 和 srl_annotations 路径 !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        parser = SRLParser()
        parser.process_flickr30k_annotations(original_annotations, srl_annotations)