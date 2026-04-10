import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from scipy.stats import linregress

def fcal_1(path: str):      
    # 返回后三个step中inter_energy_kcal的均值
    df = pd.read_csv(path)  # 使用时需要从label修改为正确的路径
    return df.iloc[-3:, 1].mean()


def fcal_2():
    pass


def extract_total_energy(folder_path, count=[0]):   
    # 返回一个列表，包含label和total_energy以及相关的各种特征
    results = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(file_path)

                # 清理列名
                df.columns = df.columns.str.strip()

                if "Total Energy (kJ/mole)" not in df.columns:
                    print(f"⚠️ {file} 缺少 Total Energy 列")
                    continue

                energy = df["Total Energy (kJ/mole)"].dropna()

                if len(energy) < 2:
                    print(f"⚠️ {file} 数据太少")
                    continue

                # ===== 基础特征 =====
                final_energy = energy.iloc[-1]
                mean_energy = energy.mean()
                std_energy = energy.std()

                # ===== 均值回归特征 =====
                final_minus_mean = final_energy - mean_energy

                # ===== 差分特征 =====
                diff = energy.diff().dropna()

                last_diff = diff.iloc[-1]
                mean_diff = diff.mean()
                std_diff = diff.std()

                # ===== label =====
                label = os.path.splitext(file)[0]

                # 单轨迹性质
                results.append({
                    "label": label,
                    "final_energy": final_energy,
                    #"mean_energy": mean_energy,
                    #"std_energy": std_energy,
                    #"final_minus_mean": final_minus_mean,
                    "last_diff": last_diff,       # 最后一步能量变化
                    "mean_diff": mean_diff,       # 每个step变化整体趋势
                    #"std_diff": std_diff
                })

            except Exception as e:
                # print(f"❌ 处理 {file} 出错: {e}")
                count[0] += 1

            # 整体性质
            result_df = pd.DataFrame(results)
            global_mean = result_df["final_energy"].mean()
            result_df["global_centered_energy"] = (result_df["final_energy"] - global_mean)
            result_df["z_score_energy"] = (
                (result_df["final_energy"] - global_mean) /
                result_df["final_energy"].std()
                )   # Z-score

            # 转回list
            results = result_df.to_dict(orient="records")


    print(f"✅ 特征已获取")
    return results

def extract_interact_energy(folder_path, count=[0]): 
    # 返回一个列表，包含相互作用能最后三个step的均值、线性回归值
    results = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            
            try:
                df = pd.read_csv(file_path)

                # 检查列是否存在
                if "inter_energy_kcal" not in df.columns:
                    # print(f"⚠️ {file} 不包含目标列")
                    continue

                # 空数值处理
                energy_series = df["inter_energy_kcal"].dropna()
                if len(energy_series) == 0:
                    print(f"⚠️ {file} 没有有效能量数据")
                    continue

                lastmean_energy = energy_series.iloc[-3:].mean()

                # 线性回归
                slope, intercept, r, p_value, std_err = linregress(range(len(energy_series)), energy_series)

                # 文件名去掉 .csv 作为 label
                label = os.path.splitext(file)[0]

                results.append({
                    "label": label,
                    "lastmean Energy": lastmean_energy,
                    "trend_slope": slope,
                    "trend_r2": r**2
                })

            except Exception as e:
                # print(f"❌ 处理 {file} 出错: {e}")
                count[0] += 1

    # 保存为新的 CSV
    # result_df = pd.DataFrame(results)
    # result_df.to_csv(output_file, index=False)
    print(f"✅ 已处理完毕")
    return results

def extract_interact_force(folder_path): 
    pass

def preprocess_total_energy(results: list, count=[0]):
    # 均值回归和差异化处理 返回包含若干特征的results列表
    # 对数值过大的特征，需要进行log或其他处理
    df = pd.DataFrame(results)
    # 自动检测所有行中数值超过1000的列（基于第一行或全局max判断）
    numeric_cols = df.select_dtypes(include='number').columns
    cols_to_normalize = [col for col in numeric_cols if df[col].abs().max() > 1000]

    print("需要归一化的列:", cols_to_normalize)
    # 输出: ['final_energy', 'last_diff', 'mean_diff']

    df["final_energy"] = -df["final_energy"] 
    # 取对数
    df["final_energy"]  = np.log(df["final_energy"])

    # Min-Max 归一化到 [0, 1]
    scaler = RobustScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    # 负数的情况下使用
    # df["final_energy"] = 1-df["final_energy"]   

    # 转回list
    results = df.to_dict(orient="records")

    return results
    

def add_to_csv(results: list, output_file: str):
    # 将list中的数据成列添加到output_file尾部
    df_new = pd.DataFrame(results)

    # 如果文件已存在 → 追加行（不会覆盖！）
    if os.path.exists(output_file):
        df_old = pd.read_csv(output_file)
        df_final = pd.merge(
            df_old,        # 旧表
            df_new,        # 新表
            on="label",    # 按 label 匹配！
            how="outer"    # 所有label都保留（不会丢数据）
        )
    else:
        df_final = df_new  # 文件不存在，直接新建

    # 保存（不乱码 + 不保留索引）
    df_final.to_csv(output_file, index=False, encoding="utf-8-sig")
    return 0



def main():  # 计算total_energy和interaction_energy特征
    results = []
    results = extract_total_energy("energylog_folder")
    results = preprocess_total_energy(results)
    add_to_csv(results, "feature_calculate/Summary_total_intera.csv")
    results = extract_interact_energy("interaction_folder")
    add_to_csv(results, "feature_calculate/Summary_total_intera.csv")

    return


if __name__ == "__main__":
    main()
    # pass
   