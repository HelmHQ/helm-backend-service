import pandas as pd
import numpy as np
import json
from scipy import stats
from datetime import datetime

class InsightEngine:
    def _init_(self, raw_data_list):
        """
        raw_data_list: List of dicts (DailyMetric objects from Flutter)
        """
        self.df = pd.DataFrame(raw_data_list)
        
    def run_analysis(self):
        """Main pipeline execution."""
        # Need a reasonable amount of data to run stats
        if len(self.df) < 5:
            return {"error": "Not enough data. Need at least 5 days of history."}
            
        self._preprocess()
        self._feature_engineering()
        
        # Run statistical tests
        correlations = self._analyze_numerical_correlations()
        group_diffs = self._analyze_categorical_differences()
        
        # Combine all findings
        all_insights = correlations + group_diffs
        
        # Sort by 'strength' (absolute value of correlation or effect size)
        all_insights.sort(key=lambda x: abs(x.get('strength', 0)), reverse=True)
        
        # Return top 5 strongest statistically significant insights
        return all_insights[:5]

    def _preprocess(self):
        # 1. Date Extraction
        if 'date' in self.df.columns:
            self.df['date_obj'] = pd.to_datetime(self.df['date'])
            self.df['day_of_week'] = self.df['date_obj'].dt.day_name()
            self.df['is_weekend'] = self.df['date_obj'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        
        # 2. Fill Missing Values
        # Numerical -> Mean
        num_cols = ['totalScreenOnTime', 'appUsageFrequency', 'deviceUnlockCount', 
                   'notificationsReceived', 'dailyStepCount', 'sleepTime', 
                   'appSwitchingFrequency', 'entropyScore',
                   'sleepQuality', 'moodScore', 'stressScore', 
                   'productivityScore', 'energyLevel']
                   
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                mean_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(0 if pd.isna(mean_val) else mean_val)

        # Categorical/Binary -> Mode
        cat_cols = ['fragmentation', 'exercise', 'dominantActivityTag']
        for col in cat_cols:
             if col in self.df.columns:
                mode_val = self.df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 0
                self.df[col] = self.df[col].fillna(fill_val)

    def _feature_engineering(self):
        self.expand_json_column('screenTimeByCategory', prefix='cat')
        self._expand_app_details()
        self._expand_sentiment()

    def _expand_json_column(self, col_name, prefix):
        def parse_safe(x):
            if isinstance(x, str):
                try: return json.loads(x)
                except: return {}
            return {}
            
        if col_name in self.df.columns:
            expanded = self.df[col_name].apply(parse_safe).apply(pd.Series)
            expanded = expanded.fillna(0).add_prefix(prefix)
            self.df = pd.concat([self.df, expanded], axis=1)

    def _expand_app_details(self):
        if 'appDetailsByCategory' not in self.df.columns: return

        app_data = []
        for _, row in self.df.iterrows():
            daily_apps = {}
            raw = row.get('appDetailsByCategory', '{}')
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                    for cat, apps in data.items():
                        if isinstance(apps, dict):
                            for app_name, duration in apps.items():
                                if duration >= 300:
                                    daily_apps[f"app_{app_name}"] = duration
                except: pass
            app_data.append(daily_apps)
            
        app_df = pd.DataFrame(app_data).fillna(0)
        self.df = pd.concat([self.df, app_df], axis=1)

    def _expand_sentiment(self):
        if 'sentimentAnalysisRaw' not in self.df.columns: return

        sentiments_list = []
        for _, row in self.df.iterrows():
            day_sentiments = {}
            raw = row.get('sentimentAnalysisRaw')
            if isinstance(raw, str) and raw:
                try:
                    data = json.loads(raw)
                    labels = data.get('sentiments', [])
                    scores = data.get('scores', [])
                    for i in range(min(2, len(labels))):
                        label = labels[i]
                        score = scores[i]
                        if label != "neutral" or len(labels) == 1:
                            day_sentiments[f"emotion_{label}"] = score
                except: pass
            sentiments_list.append(day_sentiments)
            
        sent_df = pd.DataFrame(sentiments_list).fillna(0)
        self.df = pd.concat([self.df, sent_df], axis=1)

    # --- STATISTICAL TESTS ---

    def _analyze_numerical_correlations(self):
        """Pearson Correlation for Metric vs Metric."""
        insights = []
        targets = ['sleepQuality', 'moodScore', 'stressScore', 'productivityScore', 'energyLevel']
        # Features: Screen Time Cats, Specific Apps, Emotions, Passive Data
        feature_cols = [c for c in self.df.columns if c.startswith(('cat_', 'app_', 'emotion_', 'total', 'daily', 'entropy', 'appSwitch'))]
        
        for target in targets:
            if target not in self.df.columns: continue
            for feature in feature_cols:
                if feature not in self.df.columns: continue
                if feature == target: continue 
                
                valid_data = self.df[[feature, target]].dropna()
                if len(valid_data) < 3: continue 
                if valid_data[feature].nunique() <= 1 or valid_data[target].nunique() <= 1: continue

                # Pearson Correlation
                corr, p_value = stats.pearsonr(valid_data[feature], valid_data[target])
                
                if abs(corr) > 0.4 and p_value < 0.15:
                    # --- PREPARE SCATTER PLOT DATA ---
                    # We normalize x values for cleaner plotting if needed, 
                    # but sending raw values is better for tooltip context.
                    plot_points = []
                    # Limit to last 30 points to keep payload small
                    recent_data = valid_data.tail(30)
                    for _, row in recent_data.iterrows():
                        plot_points.append({"x": float(row[feature]), "y": float(row[target])})

                    insights.append({
                        "type": "correlation",
                        "feature": feature,
                        "target": target,
                        "strength": round(corr, 2),
                        "p_value": round(p_value, 3),
                        "message": f"Strong {'positive' if corr > 0 else 'negative'} link found.",
                        "chart_data": {
                            "type": "scatter",
                            "points": plot_points,
                            "x_label": feature.replace('cat_', '').replace('app_', '').replace('_', ' ').title(),
                            "y_label": target
                        }
                    })
        return insights

    def _analyze_categorical_differences(self):
        """T-Test (Binary) and ANOVA (Multi-class)."""
        insights = []
        targets = ['sleepQuality', 'moodScore', 'stressScore', 'productivityScore', 'energyLevel']
        binary_cols = ['is_weekend', 'exercise', 'fragmentation']
        
        for target in targets:
            if target not in self.df.columns: continue
            for col in binary_cols:
                if col not in self.df.columns: continue
                
                group0 = self.df[self.df[col] == 0][target]
                group1 = self.df[self.df[col] == 1][target]
                
                if len(group0) > 1 and len(group1) > 1:
                    t_stat, p_value = stats.ttest_ind(group0, group1)
                    if p_value < 0.15: 
                        diff = group1.mean() - group0.mean()
                        
                        # --- PREPARE BAR CHART DATA ---
                        bar_data = [
                            {"label": f"No {col.capitalize()}", "value": round(group0.mean(), 2)},
                            {"label": f"{col.capitalize()}", "value": round(group1.mean(), 2)}
                        ]

                        insights.append({
                            "type": "t_test",
                            "feature": col,
                            "target": target,
                            "strength": round(diff, 2),
                            "p_value": round(p_value, 3),
                            "message": f"Significant difference when {col} happens.",
                            "chart_data": {
                                "type": "bar",
                                "bars": bar_data,
                                "y_label": target
                            }
                        })

        if 'dominantActivityTag' in self.df.columns:
            for target in targets:
                groups_map = {name: group[target].dropna().values for name, group in self.df.groupby('dominantActivityTag')}
                valid_groups = [vals for vals in groups_map.values() if len(vals) > 1]
                
                if len(valid_groups) > 1:
                    f_stat, p_value = stats.f_oneway(*valid_groups)
                    if p_value < 0.15:
                         means = self.df.groupby('dominantActivityTag')[target].mean()
                         best_activity = means.idxmax()
                         
                         # --- PREPARE BAR CHART DATA (Top 3 activities) ---
                         sorted_means = means.sort_values(ascending=False).head(4)
                         bar_data = [{"label": act, "value": round(val, 2)} for act, val in sorted_means.items()]

                         insights.append({
                            "type": "anova",
                            "feature": "dominantActivityTag",
                            "target": target,
                            "strength": round(f_stat, 2),
                            "p_value": round(p_value, 3),
                            "detail": f"Highest with {best_activity}",
                            "chart_data": {
                                "type": "bar",
                                "bars": bar_data,
                                "y_label": target
                            }
                        })
                        
        return insights