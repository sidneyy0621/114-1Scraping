import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import os

# 設定繁體中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. 讀取資料
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Customer Data', 'new_customer_data.csv')
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 設定輸出目錄
    output_dir = os.path.join(script_dir, 'Q5_ans')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(file_path)
    
    # 2. 地理位置分群 (K-Means)
    # 使用 緯度, 經度
    X = df[['緯度', '經度']]
    
    # 標準化 (Normalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 決定群數，這裡設為 5
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) #紅色叉叉(質心)
    # 使用標準化後的資料進行分群
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    print(f"已將顧客分為 {n_clusters} 群 (Cluster 0 to {n_clusters-1})")
    
    # 繪製分群結果
    plt.figure(figsize=(10, 6))
    
    # 使用 Seaborn 繪製，並強制將 Cluster 視為類別變數以產生離散圖例
    # hue_order 確保圖例按照 0, 1, 2, 3, 4 順序排列
    sns.scatterplot(x=df['經度'], y=df['緯度'], hue=df['Cluster'], 
                    palette='viridis', s=50, alpha=0.6, legend='full')
    
    # 標註群組中心點
    centers = kmeans.cluster_centers_
    # 將標準化後的中心點還原回原始經緯度
    centers_inverse = scaler.inverse_transform(centers)
    # 注意：inverse_transform 回傳的是 [緯度, 經度] (因為 X 是這樣建的)，但畫圖是 x=經度, y=緯度
    plt.scatter(centers_inverse[:, 1], centers_inverse[:, 0], c='red', s=200, marker='X', label='群組中心')
    
    plt.title('客戶地理位置分群 (K-Means Clustering)')
    plt.xlabel('經度 (Longitude)')
    plt.ylabel('緯度 (Latitude)')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    
    output_plot_path = os.path.join(output_dir, 'Q5_geo_clusters.png')
    plt.savefig(output_plot_path)
    plt.close()
    print(f"分群圖表已儲存至 {output_plot_path}")

    # 互動式地圖 (Plotly)
    try:
        fig = px.scatter_mapbox(df, lat="緯度", lon="經度", color="Cluster",
                                color_continuous_scale=px.colors.sequential.Viridis,
                                size_max=15, zoom=5,
                                mapbox_style="carto-positron",
                                title="顧客地理位置分群 (Interactive Map)")
        output_html_path = os.path.join(output_dir, 'Q5_geo_clusters_interactive.html')
        fig.write_html(output_html_path)
        print(f"互動式地圖已儲存至 {output_html_path}")
    except Exception as e:
        print(f"無法建立互動式地圖 (可能是缺少 plotly 套件): {e}")
    
    # 3. 比較不同群組特徵
    # 特徵: 性別, 年齡, 婚姻, 扶養人數
    print("\n--- 各群組特徵比較 ---")
    
    # 數值型特徵平均值 (年齡, 扶養人數)
    numeric_cols = ['年齡', '扶養人數', '每月費用', '總費用', '推薦次數', '平均下載量( GB)']
    print(df.groupby('Cluster')[numeric_cols].mean())
    
    # 類別型特徵分佈 (性別, 婚姻)
    for col in ['性別', '婚姻']:
        print(f"\n{col} 分佈 (百分比):")
        print(pd.crosstab(df['Cluster'], df[col], normalize='index') * 100)

    # 匯出群組特徵分析報告
    stats_numeric = df.groupby('Cluster')[numeric_cols].mean()
    
    def female_ratio(x):
        return (x == 'Female').mean()

    def churn_ratio(x):
        # 假設 '客戶狀態' 為 'Churned' 代表流失
        return (x == 'Churned').mean()

    def married_ratio(x):
        return (x == 'Yes').mean()
        
    stats_gender = df.groupby('Cluster')['性別'].agg(女性比例=female_ratio)
    stats_churn = df.groupby('Cluster')['客戶狀態'].agg(流失率=churn_ratio)
    stats_marriage = df.groupby('Cluster')['婚姻'].agg(已婚比例=married_ratio)
    
    # 計算合約類型的眾數 (Mode)
    stats_contract = df.groupby('Cluster')['合約類型'].agg(lambda x: x.mode()[0]).rename('主要合約類型')

    final_cluster_stats = pd.concat([stats_numeric, stats_gender, stats_marriage, stats_churn, stats_contract], axis=1)
    output_profile_path = os.path.join(output_dir, 'Q5_cluster_profile.csv')
    final_cluster_stats.to_csv(output_profile_path, encoding='utf-8-sig')
    print(f"群組特徵分析已儲存至 {output_profile_path}")

    # --- 加分題：視覺化群組特徵 (流失率、下載量、合約類型、推薦次數) ---
    print("正在繪製群組特徵比較圖 (流失率、下載量、合約類型、推薦次數)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. 流失率比較
    churn_rate = df.groupby('Cluster')['客戶狀態'].apply(lambda x: (x == 'Churned').mean() * 100)
    sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=axes[0, 0], palette='viridis', hue=churn_rate.index, legend=False)
    axes[0, 0].set_title('流失率比較')
    axes[0, 0].set_ylabel('流失率 (%)')
    axes[0, 0].set_xlabel('地理群組')
    for i, v in enumerate(churn_rate):
        axes[0, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center')

    # 2. 平均下載量比較
    avg_download = df.groupby('Cluster')['平均下載量( GB)'].mean()
    sns.barplot(x=avg_download.index, y=avg_download.values, ax=axes[0, 1], palette='viridis', hue=avg_download.index, legend=False)
    axes[0, 1].set_title('平均下載量比較')
    axes[0, 1].set_ylabel('平均下載量 (GB)')
    axes[0, 1].set_xlabel('地理群組')
    for i, v in enumerate(avg_download):
        axes[0, 1].text(i, v + 0.5, f'{v:.1f}', ha='center')

    # 3. 平均推薦次數比較
    avg_referrals = df.groupby('Cluster')['推薦次數'].mean()
    sns.barplot(x=avg_referrals.index, y=avg_referrals.values, ax=axes[1, 0], palette='viridis', hue=avg_referrals.index, legend=False)
    axes[1, 0].set_title('平均推薦次數比較')
    axes[1, 0].set_ylabel('平均推薦次數')
    axes[1, 0].set_xlabel('地理群組')
    for i, v in enumerate(avg_referrals):
        axes[1, 0].text(i, v + 0.1, f'{v:.1f}', ha='center')

    # 4. 合約類型分布 (堆疊長條圖)
    contract_dist = pd.crosstab(df['Cluster'], df['合約類型'], normalize='index') * 100
    contract_dist.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='viridis')
    axes[1, 1].set_title('合約類型分布')
    axes[1, 1].set_ylabel('百分比 (%)')
    axes[1, 1].set_xlabel('地理群組')
    axes[1, 1].legend(title='合約類型', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].tick_params(axis='x', rotation=0)

    output_comparison_path = os.path.join(output_dir, 'Q5_cluster_comparison_advanced.png')
    plt.savefig(output_comparison_path, bbox_inches='tight')
    plt.close()
    print(f"進階群組比較圖已儲存至 {output_comparison_path}")

    # --- 加分題 1: 手肘法 (Elbow Method) 驗證 k=5 的合理性 ---
    print("正在繪製手肘法圖表...")
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        # 注意：這裡要用標準化後的 X_scaled 來算
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o', linestyle='--')
    plt.title('Elbow Method (手肘法) - 決定最佳群數')
    plt.xlabel('群數 (k)')
    plt.ylabel('SSE (誤差平方和)')
    plt.grid(True)
    # 標記出我們選擇的 k=5
    plt.axvline(x=5, color='r', linestyle=':', label='Selected k=5')
    plt.legend()
    output_elbow_path = os.path.join(output_dir, 'Q5_elbow_method_proof.png')
    plt.savefig(output_elbow_path)
    plt.close()
    print(f"Elbow Method 圖表已儲存至 {output_elbow_path}")

    # 4. 針對其中一群組建立關聯規則
    target_cluster = 0
    print(f"\n--- 針對 Cluster {target_cluster} 建立關聯規則 ---")
    
    cluster_df = df[df['Cluster'] == target_cluster].copy()
    
    # 選擇服務相關欄位
    # 確保只包含公司提供的服務，不包含個人特徵(如性別)或合約資訊(如付款方式)，以符合題目(b)要求
    service_cols = [
        '電話服務', '多線路服務', '網路服務', '線上安全服務', 
        '線上備份服務', '設備保護計劃', '技術支援計劃', 
        '電視節目', '電影節目', '音樂節目', '無限資料下載'
    ]
    
    # 資料預處理：將 Yes/No 轉換為 True/False，或是 One-hot encoding
    # 這裡假設欄位值多為 Yes/No/No internet service 等
    # 為了簡化，我們只看 "Yes" 的項目
    
    transactions = []
    for _, row in cluster_df.iterrows():
        transaction = []
        for col in service_cols:
            if row[col] == 'Yes':
                transaction.append(col)
            # 對於網路服務，值可能是 'DSL', 'Fiber Optic', 'No'
            elif col == '網路服務' and row[col] not in ['No', 'No internet service']:
                 transaction.append(f"網路服務_{row[col]}")
        transactions.append(transaction)
        
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apriori
    min_support = 0.2
    frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
    
    # 如果找不到規則，嘗試降低 min_support
    if frequent_itemsets.empty:
        print(f"未找到頻繁項目集 (min_support={min_support})，嘗試降低至 0.05...")
        min_support = 0.05
        frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        print(f"仍未找到頻繁項目集 (min_support={min_support})")
    else:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        
        # 額外篩選 Lift > 1.0 的規則 (確保正相關)
        rules = rules[rules['lift'] > 1.0]
        
        print(f"找到 {len(rules)} 條規則 (min_support={min_support}, min_confidence=0.6, lift>1.0)")
        if not rules.empty:
            # 顯示前 10 條規則，按 lift 排序
            print(rules.sort_values('lift', ascending=False).head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            
            # 清理 frozenset 格式並重新命名欄位
            def clean_set_str(x):
                return ', '.join(list(x))

            rules['antecedents'] = rules['antecedents'].apply(clean_set_str)
            rules['consequents'] = rules['consequents'].apply(clean_set_str)

            rules = rules.rename(columns={
                'antecedents': '前項(購買A)',
                'consequents': '後項(購買B)',
                'support': '支持度',
                'confidence': '信賴度',
                'lift': '提升度'
            })

            # 儲存規則到檔案
            output_rules_path = os.path.join(output_dir, 'Q5_association_rules_cleaned.csv')
            rules.to_csv(output_rules_path, index=False, encoding='utf-8-sig')
            print(f"關聯規則已儲存至 {output_rules_path}")

            # 繪製關聯規則散佈圖 (Support vs Confidence, color=Lift)
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(rules['支持度'], rules['信賴度'], 
                                c=rules['提升度'], cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter, label='提升度 (Lift)')
            plt.title(f'關聯規則散佈圖 (Cluster {target_cluster})')
            plt.xlabel('支持度 (Support)')
            plt.ylabel('信賴度 (Confidence)')
            plt.grid(True, alpha=0.3)
            
            output_scatter_path = os.path.join(output_dir, 'Q5_association_rules_scatter.png')
            plt.savefig(output_scatter_path)
            plt.close()
            print(f"關聯規則散佈圖已儲存至 {output_scatter_path}")

            # --- 加分題：Top 5 精選商業規則長條圖 (手動挑選具代表性規則) ---
            print("正在繪製 Top 5 精選商業規則長條圖...")
            
            # 定義我們想要找的規則 (前項 -> 後項)
            target_rules_spec = [
                ({'電影節目'}, {'音樂節目'}, '影音雙享 (互補)'),
                ({'音樂節目'}, {'電影節目'}, '影音雙享 (互補)'),
                ({'無限資料下載'}, {'網路服務'}, '數位綁定 (基礎)'),
                ({'電影節目'}, {'網路服務'}, '數位綁定 (基礎)'),
                ({'多線路服務'}, {'電話服務'}, '家庭通訊 (依賴)')
            ]
            
            selected_rules = []
            
            # 依照指定的順序搜尋規則 (確保圖表順序符合報告邏輯：由上到下)
            for target_ant, target_con, label in target_rules_spec:
                found = False
                for idx, row in rules.iterrows():
                    ant = set(row['前項(購買A)'].split(', '))
                    con = set(row['後項(購買B)'].split(', '))
                    
                    if ant == target_ant and con == target_con:
                        # 找到匹配的規則，加入列表
                        selected_rules.append({
                            'rule': f"若 [{', '.join(ant)}] \n-> 推薦 [{', '.join(con)}]",
                            'lift': row['提升度'],
                            'confidence': row['信賴度'],
                            'type': label
                        })
                        found = True
                        break
                if not found:
                    print(f"Warning: 未找到規則 {target_ant} -> {target_con}")
            
            # 轉為 DataFrame
            if selected_rules:
                top_5_rules = pd.DataFrame(selected_rules)
            else:
                # 如果找不到(防呆)，還是退回用 Lift 排序
                print("未找到指定規則，改用 Lift 排序...")
                top_5_rules = rules.sort_values(by='提升度', ascending=False).head(5)
                top_5_rules['rule'] = top_5_rules.apply(lambda x: f"若 [{x['前項(購買A)']}] \n-> 推薦 [{x['後項(購買B)']}]", axis=1)
                top_5_rules['lift'] = top_5_rules['提升度']

            plt.figure(figsize=(10, 6))
            # 繪製水平長條圖
            y_pos = range(len(top_5_rules))
            bars = plt.barh(y_pos, top_5_rules['lift'], color='skyblue', alpha=0.8)
            plt.yticks(y_pos, top_5_rules['rule'], fontsize=10)
            plt.xlabel('提升度 (Lift)')
            plt.title(f'Top 5 精選商業洞察規則 (Cluster {target_cluster})')
            plt.gca().invert_yaxis()  # 讓第一條在上面
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            
            # 在 Bar 旁標註信賴度
            for i, bar in enumerate(bars):
                conf = top_5_rules.iloc[i]['confidence'] if 'confidence' in top_5_rules.columns else 0
                plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                         f' Lift: {bar.get_width():.2f}\n (Conf: {conf:.0%})', 
                         va='center', fontsize=9, color='black')
            
            output_bar_path = os.path.join(output_dir, 'Q5_top5_rules_barchart.png')
            plt.savefig(output_bar_path, bbox_inches='tight')
            plt.close()
            print(f"Top 5 精選規則長條圖已儲存至 {output_bar_path}")

if __name__ == "__main__":
    main()