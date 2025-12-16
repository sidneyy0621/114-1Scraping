import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import os
import numpy as np

# ================= 1. 設定區 =================
input_file = r'c:\Users\Chen\Desktop\大數據\midterm\Customer Data\new_customer_data.csv'
output_dir = r'c:\Users\Chen\Desktop\大數據\midterm\Q6_ans'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# 定義常數
AGE_ORDER = ['Young', 'Middle', 'Old']
COLORS_DICT = {'Young': '#FF9999', 'Middle': '#66B2FF', 'Old': '#99FF99'}
COLORS_LIST = ['#FF9999', '#66B2FF', '#99FF99', '#D3D3D3'] # Young, Middle, Old, Excluded

# ================= 2. 資料讀取與前處理 =================
try:
    df = pd.read_csv(input_file, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(input_file, encoding='big5')
    print("已使用 Big5 編碼讀取。")

# 定義年齡分組
def categorize_age(age):
    if age >= 60: return 'Old'
    elif 31 <= age <= 59: return 'Middle'
    elif 15 <= age <= 30: return 'Young'
    else: return '0-14' # 0-14 歲

df['AgeGroup'] = df['年齡'].apply(categorize_age)

# 定義二元欄位
binary_cols = ['電話服務', '多線路服務', '網路服務', '線上安全服務', '線上備份服務', 
               '設備保護計劃', '技術支援計劃', '電視節目', '電影節目', '音樂節目', '無限資料下載']
internet_types = ['Cable', 'Fiber Optic', 'DSL']

# 預先過濾資料 (供後續分析使用)
df_filtered = df.dropna(subset=['AgeGroup']).copy()

# ================= 3. 年齡分佈圖 (包含區間外) =================
# 準備繪圖數據 (將 None 填補為 '0-14 (Excluded)')
age_counts = df['AgeGroup'].fillna('0-14 (Excluded)').value_counts()
# 確保順序好看
order = AGE_ORDER + ['0-14 (Excluded)']
# 過濾掉不存在的 key (防止報錯)
order = [o for o in order if o in age_counts.index]

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=age_counts[order].index, y=age_counts[order].values, palette=COLORS_LIST)

# 在柱狀圖上標示數值
for i, v in enumerate(age_counts[order].values):
    ax.text(i, v + 5, str(v), ha='center', fontsize=12, fontweight='bold')

plt.title('各年齡層樣本數分佈 (含排除區間)', fontsize=16)
plt.ylabel('人數', fontsize=12)
plt.xlabel('年齡群組', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

dist_path = os.path.join(output_dir, 'age_distribution.png')
plt.savefig(dist_path, dpi=150)

# ================= 4. 服務持有率熱力圖 =================
# 使用已過濾的資料
df_heatmap = df_filtered.copy()

# 將 Yes/No 轉換為 1/0 以計算平均值 (即持有率)
for col in binary_cols:
    df_heatmap[col] = df_heatmap[col].map({'Yes': 1, 'No': 0}).fillna(0)

# 計算各族群的平均持有率
penetration = df_heatmap.groupby('AgeGroup')[binary_cols].mean()
# 調整列順序 (Index)
penetration = penetration.reindex(AGE_ORDER)

plt.figure(figsize=(12, 5))
sns.heatmap(penetration, annot=True, fmt=".0%", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': '持有率'})
plt.title('各年齡層服務持有率 (Service Penetration)', fontsize=16)
plt.ylabel('年齡群組', fontsize=12)
plt.xlabel('服務項目', fontsize=12)
plt.xticks(rotation=45, ha='right')

heatmap_path = os.path.join(output_dir, 'service_penetration_heatmap.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')

# ================= 4.1 服務持有率雷達圖 =================
# 準備數據 (使用簡稱以優化顯示)
radar_data = penetration.copy()
name_mapping = {
    '電話服務': '電話', '多線路服務': '多線路', '網路服務': '網路',
    '線上安全服務': '線上安全', '線上備份服務': '線上備份', '設備保護計劃': '設備保護',
    '技術支援計劃': '技術支援', '電視節目': '電視', '電影節目': '電影',
    '音樂節目': '音樂', '無限資料下載': '無限資料'
}
radar_data.columns = [name_mapping.get(c, c) for c in radar_data.columns]

categories = list(radar_data.columns)
N = len(categories)

# 設定角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # 閉合

# 初始化圖表
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# 設定第一軸在正上方
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 設定 X 軸標籤
plt.xticks(angles[:-1], categories, size=12)

# 設定 Y 軸標籤
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=10)
plt.ylim(0, 1)

# 繪製每個族群
for group in AGE_ORDER:
    if group in radar_data.index:
        values = radar_data.loc[group].values.flatten().tolist()
        values += values[:1] # 閉合
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=group, color=COLORS_DICT[group], marker='o')
        ax.fill(angles, values, color=COLORS_DICT[group], alpha=0.25)

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.title('各年齡層服務持有率雷達圖', size=20, y=1.08)

radar_path = os.path.join(output_dir, 'service_penetration_radar.png')
plt.savefig(radar_path, dpi=150, bbox_inches='tight')

# ================= 5. 關聯規則分析 =================

# df_filtered 已經在前面定義過了
all_rules_data = []
summary_report = ["=== Q6 關聯規則分析摘要報告 (含圖表解讀建議) ===\n"]

# 冗餘過濾函數
def remove_redundant_rules(rules_df, tolerance=0.01):
    if rules_df.empty: return rules_df
    rules_df['length'] = rules_df['antecedents'].apply(len) + rules_df['consequents'].apply(len)
    sorted_df = rules_df.sort_values(by=['lift', 'length', 'confidence'], ascending=[False, True, False])
    final_rules = []
    for idx, row in sorted_df.iterrows():
        is_redundant = False
        for kept in final_rules:
            if abs(row['lift'] - kept['lift']) <= tolerance:
                if kept['antecedents'].issubset(row['antecedents']) and kept['consequents'].issubset(row['consequents']):
                    is_redundant = True
                    break
        if not is_redundant:
            final_rules.append(row)
    return pd.DataFrame(final_rules).drop(columns=['length'])

for group in AGE_ORDER:
    group_data = df_filtered[df_filtered['AgeGroup'] == group]
    if group_data.empty: continue
        
    # One-hot Encoding
    basket = group_data[binary_cols].replace({'Yes': True, 'No': False}).fillna(False)
    internet_dummies = pd.get_dummies(group_data['網路連線類型'], prefix='網路類型')
    target_internet_cols = [f'網路類型_{t}' for t in internet_types]
    for col in target_internet_cols:
        if col in internet_dummies.columns:
            basket[col] = internet_dummies[col].astype(bool)
        else:
            basket[col] = False 
            
    try:
        # 提高支持度門檻 (0.1 -> 0.12)
        frequent_itemsets = apriori(basket, min_support=0.12, use_colnames=True)
        if frequent_itemsets.empty: continue
        # 提高信賴度門檻 (0.5 -> 0.6)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        
        # 提高 Lift 門檻 (1.0 -> 1.2) 確保強相關
        rules = rules[rules['lift'] > 1.2]
        
        # 基礎過濾
        def is_insightful(row):
            cons = row['consequents']
            if len(cons) == 1 and '網路服務' in cons and row['confidence'] > 0.95:
                return False
            return True
        refined_rules = rules[rules.apply(is_insightful, axis=1)].copy()
        
        # 進階過濾
        final_clean_rules = remove_redundant_rules(refined_rules)
        
        # 限制最終輸出數量 (只保留 Lift 最高的 20 筆)
        if len(final_clean_rules) > 20:
            final_clean_rules = final_clean_rules.head(20)
        
        # 儲存 CSV
        csv_path = os.path.join(output_dir, f'rules_{group}_final.csv')
        final_clean_rules.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        final_clean_rules['Group'] = group
        all_rules_data.append(final_clean_rules)
        
        # 文字報告
        summary_report.append(f"\n【族群: {group}】 (Top 5 精選規則)")
        top_5 = final_clean_rules.head(5)
        for idx, row in top_5.iterrows():
            ant = ", ".join(list(row['antecedents']))
            con = ", ".join(list(row['consequents']))
            lift = round(row['lift'], 2)
            summary_report.append(f"  * 若有 [{ant}] -> 推薦 [{con}] (Lift: {lift})")

    except Exception as e:
        print(f"  Error {group}: {e}")

# ================= 6. 輸出報告與規則散佈圖 =================

# 輸出文字報告
report_path = os.path.join(output_dir, 'analysis_summary_final.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(summary_report))

# 輸出規則散佈圖
if all_rules_data:
    all_df = pd.concat(all_rules_data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for i, group in enumerate(AGE_ORDER):
        ax = axes[i]
        subset = all_df[all_df['Group'] == group]
        if not subset.empty:
            plot_data = subset.head(50)
            sns.scatterplot(
                data=plot_data, x='support', y='lift', size='confidence', 
                sizes=(50, 250), alpha=0.7, color=COLORS_DICT.get(group, 'gray'), ax=ax, legend=False
            )
            top = plot_data.iloc[0]
            ax.annotate('Top 1', xy=(top['support'], top['lift']), 
                        xytext=(top['support'], top['lift']+0.5),
                        arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

        ax.set_title(f'{group} (Top 50 Rules)', fontsize=14)
        ax.set_xlabel('Support')
        if i == 0: ax.set_ylabel('Lift')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rules_visualization_final.png')
    plt.savefig(plot_path, dpi=150)

    # ================= 6.1 Top 5 關聯規則長條圖 (依 Lift 排序) =================
    # 設定圖表大小 (3個子圖，垂直排列)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    if len(AGE_ORDER) == 1: axes = [axes] # 防呆
    
    for i, group in enumerate(AGE_ORDER):
        ax = axes[i]
        # 找出該族群的規則資料
        group_rules = next((df for df in all_rules_data if not df.empty and df['Group'].iloc[0] == group), pd.DataFrame())
        
        if not group_rules.empty:
            # 取 Top 5 (已在前面排序過，但保險起見再排一次)
            top_5_rules = group_rules.sort_values(by='lift', ascending=False).head(5)
            
            # 製作標籤
            labels = []
            for _, row in top_5_rules.iterrows():
                ant = ", ".join(list(row['antecedents']))
                con = ", ".join(list(row['consequents']))
                # 簡化標籤
                labels.append(f"若 [{ant}] \n-> 推薦 [{con}]")
            
            # 繪製水平長條圖
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, top_5_rules['lift'], color=COLORS_DICT.get(group, 'gray'), alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=12)
            ax.invert_yaxis()  # 讓最高的在上面
            
            # 標示數值
            for j, v in enumerate(top_5_rules['lift']):
                ax.text(v, j, f' {v:.2f}', va='center', fontweight='bold')
                
            ax.set_title(f'{group} 族群 - Top 5 關聯規則 (依 Lift 排序)', fontsize=16)
            ax.set_xlabel('Lift Score', fontsize=12)
            # 自動調整 x 軸範圍，讓長條圖好看一點
            min_lift = top_5_rules['lift'].min()
            ax.set_xlim(left=max(0, min_lift - 0.5)) 
            ax.grid(axis='x', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, '無足夠規則', ha='center', va='center', fontsize=14)
            ax.set_title(f'{group} 族群', fontsize=16)

    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, 'top5_rules_barchart.png')
    plt.savefig(bar_chart_path, dpi=150)