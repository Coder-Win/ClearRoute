import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

class TreeCompatibleMoE:
    def __init__(self, n_experts=3):
        self.n_experts = n_experts
        
        # Gating Network utilizing multi-core CPU
        self.gating_network = XGBClassifier(
            n_estimators=150, 
            learning_rate=0.05, 
            max_depth=6, 
            objective='multi:softprob', 
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
        self.experts = []
        
        # Expert Hyperparameters utilizing multi-core CPU
        self.expert_params = {
            'learning_rate': 0.015, 
            'max_depth': 8, 
            'objective': 'survival:aft', 
            'eval_metric': 'aft-nloglik', 
            'tree_method': 'hist',
            'seed': 42,
            'nthread': -1
        }

    def fit(self, X_train, y_train, censor_train, sample_weight=None):
        y_log = np.log1p(y_train).values.reshape(-1, 1)
        # K-Means elegantly handles the power-law skew in log-space
        kmeans = KMeans(n_clusters=self.n_experts, random_state=42, n_init=10)
        raw_labels = kmeans.fit_predict(y_log)
        
        # Sort clusters so Expert 0 handles the shortest durations and Expert N handles the longest
        sorted_centers = np.argsort(kmeans.cluster_centers_.flatten())
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_centers)}
        expert_labels = np.vectorize(label_mapping.get)(raw_labels)
        
        # Train the router 
        self.gating_network.fit(X_train, expert_labels, sample_weight=sample_weight)

        for k in range(self.n_experts):
            mask = (expert_labels == k)
            X_k = X_train[mask]
            y_k = y_train[mask]
            censor_k = censor_train[mask]
            
            weight_k = sample_weight[mask] if sample_weight is not None else None

            y_lower = y_k.values
            y_upper = np.where(censor_k.values == 1, y_k.values, np.inf)

            dtrain_k = xgb.DMatrix(X_k, weight=weight_k)
            dtrain_k.set_float_info('label_lower_bound', y_lower)
            dtrain_k.set_float_info('label_upper_bound', y_upper)

            expert_model = xgb.train(self.expert_params, dtrain_k, num_boost_round=200)
            self.experts.append(expert_model)

    def predict(self, X_test):
        # 1. Router generates probabilities
        gating_probs = self.gating_network.predict_proba(X_test)
        expert_preds = np.zeros((X_test.shape[0], self.n_experts))
        dtest = xgb.DMatrix(X_test)
        
        for k, expert in enumerate(self.experts):
            # EXTREME TAIL CAP: Prevent the raw AFT from hallucinating infinite values.
            raw_pred = expert.predict(dtest)
            expert_preds[:, k] = np.clip(raw_pred, 1.0, 43200.0)

        # LOG-SPACE BLENDING: Stabilizes RMSE against tail variance
        log_preds = np.log1p(expert_preds)
        final_log_predictions = np.sum(gating_probs * log_preds, axis=1)
        final_predictions = np.expm1(final_log_predictions)
        
        return final_predictions