import pandas as pd
import re
import io

# 原始数据字符串
raw_data = """
LLM4CP
30,Prev:12,Pred:2,MSE: 3.3213603956028237e-07,-14.399438,NMSE:0.036312513053417206,SGCS:0.9076167345046997,Flops:7371229696.0,Params:6975548.0

LLM4CP
30,Prev:12,Pred:4,MSE: 6.595129207198624e-07,-11.422166,NMSE:0.07207481563091278,SGCS:0.8487581014633179,Flops:7374393856.0,Params:6975574.0

LLM4CP
30,Prev:12,Pred:8,MSE: 1.3582603060058318e-06,-8.284568,NMSE:0.14843736588954926,SGCS:0.7232195138931274,Flops:7380722176.0,Params:6975626.0

LLM4CP
60,Prev:12,Pred:2,MSE: 1.3861172192264348e-06,-8.230956,NMSE:0.15028110146522522,SGCS:0.7158355712890625,Flops:7371229696.0,Params:6975548.0

LLM4CP
60,Prev:12,Pred:4,MSE: 1.4879681202728534e-06,-7.9222765,NMSE:0.1613512635231018,SGCS:0.6927688717842102,Flops:7374393856.0,Params:6975574.0

LLM4CP
60,Prev:12,Pred:8,MSE: 1.6886416460693e-06,-7.3728375,NMSE:0.1831117570400238,SGCS:0.6661918759346008,Flops:7380722176.0,Params:6975626.0

LLM4CP
120,Prev:12,Pred:2,MSE: 1.435743683941837e-06,-7.8840866,NMSE:0.162776380777359,SGCS:0.687012791633606,Flops:7371229696.0,Params:6975548.0

LLM4CP
120,Prev:12,Pred:4,MSE: 1.4769678955417476e-06,-7.7639236,NMSE:0.16734303534030914,SGCS:0.6781250238418579,Flops:7374393856.0,Params:6975574.0

LLM4CP
120,Prev:12,Pred:8,MSE: 1.6772030448919395e-06,-7.2117777,NMSE:0.190030038356781,SGCS:0.6577504873275757,Flops:7380722176.0,Params:6975626.0

LLM4CP
x,Prev:12,Pred:2,MSE: 1.191188516713737e-06,-8.72476,NMSE:0.13412941992282867,SGCS:0.748817503452301,Flops:7371229696.0,Params:6975548.0

LLM4CP
x,Prev:12,Pred:4,MSE: 1.3589443597084028e-06,-8.1523075,NMSE:0.15302741527557373,SGCS:0.707690954208374,Flops:7374393856.0,Params:6975574.0

LLM4CP
x,Prev:12,Pred:8,MSE: 1.5806122064532246e-06,-7.496071,NMSE:0.17798888683319092,SGCS:0.6803868412971497,Flops:7380722176.0,Params:6975626.0
"""

def parse_metrics_to_excel(text_data, output_file="lstm_results.xlsx"):
    lines = text_data.strip().split('\n')
    parsed_data = []
    
    current_model = "result/p.csv"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 判断是模型名称行还是数据行
        # 数据行通常包含逗号和具体的指标关键词
        if ',' not in line and 'Prev:' not in line:
            current_model = line
        else:
            # 这是一个数据行
            # 格式示例: 30,Prev:12,Pred:2,MSE: 1.6e-06,-7.4,NMSE:0.17...
            
            record = {'Model': current_model}
            
            # 1. 提取 Velocity (速度)，位于第一个逗号前
            parts = line.split(',', 1) # 只分割第一个逗号，因为后面MSE里也有逗号
            velocity_str = parts[0].strip()
            record['Velocity'] = velocity_str
            
            rest_of_line = parts[1]
            
            # 2. 使用正则表达式提取各个指标
            # 提取 Prev
            prev_match = re.search(r'Prev:(\d+)', rest_of_line)
            if prev_match: record['Prev'] = int(prev_match.group(1))
            
            # 提取 Pred
            pred_match = re.search(r'Pred:(\d+)', rest_of_line)
            if pred_match: record['Pred'] = int(pred_match.group(1))
            
            # 提取 MSE
            # MSE后面跟着两个数字，中间有逗号，我们只取第一个科学计数法的数值
            mse_match = re.search(r'MSE:\s*([0-9\.eE\-\+]+)', rest_of_line)
            if mse_match: record['MSE'] = float(mse_match.group(1))
            
            # 提取 NMSE
            nmse_match = re.search(r'NMSE:([0-9\.]+)', rest_of_line)
            if nmse_match: record['NMSE'] = float(nmse_match.group(1))
            
            # 提取 SGCS
            sgcs_match = re.search(r'SGCS:([0-9\.]+)', rest_of_line)
            if sgcs_match: record['SGCS'] = float(sgcs_match.group(1))
            
            # 提取 Flops
            flops_match = re.search(r'Flops:([0-9\.]+)', rest_of_line)
            if flops_match: record['Flops'] = float(flops_match.group(1))
            
            # 提取 Params
            params_match = re.search(r'Params:([0-9\.]+)', rest_of_line)
            if params_match: record['Params'] = float(params_match.group(1))
            
            parsed_data.append(record)
            
    # 创建 DataFrame
    df = pd.DataFrame(parsed_data)
    
    # 调整列顺序，使其更符合阅读习惯
    cols_order = ['Model', 'Velocity', 'Prev', 'Pred', 'MSE', 'NMSE', 'SGCS', 'Flops', 'Params']
    # 确保列都存在，防止正则没匹配到报错
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order]
    
    # 导出到 Excel
    df.to_excel(output_file, index=False)
    print(f"转换成功！文件已保存为: {output_file}")
    print("前5行预览：")
    print(df.head())

if __name__ == "__main__":
    parse_metrics_to_excel(raw_data)