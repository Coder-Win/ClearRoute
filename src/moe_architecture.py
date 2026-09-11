import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.special import softmax

class TreeCompatibleMoE:
    def __init__(self, n_experts=3, temperature=1.5):
        self.n_experts = n_experts
        self.temperature = temperature
        
        # Gating Network (Router)
        self.gating_network = XGBClassifier(
            n_estimators=200, 
            learning_rate=0.03, 
            max_depth=6, 
            objective='multi:softprob', 
            num_class=self.n_experts, # Explicitly prevents the class inference crash
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
        self.experts = []
        
        self.base_expert_params = {
            'objective': 'survival:aft', 
            'eval_metric': 'aft-nloglik', 
            'tree_method': 'hist',
            'seed': 42,
            'nthread': -1
        }

    def fit(self, X_train, y_train, censor_train, sample_weight=None):
        print("  -> Phase 1: Training Domain Experts on Overlapping Regimes...")
        
        masks = [
            (y_train <= 90),                       # Expert 0: Routine
            (y_train >= 45) & (y_train <= 300),    # Expert 1: Moderate
            (y_train >= 180)                       # Expert 2: Severe
        ]

        # 1. Train Experts
        for k in range(self.n_experts):
            mask = masks[k]
            X_k = X_train[mask]
            y_k = y_train[mask]
            censor_k = censor_train[mask]
            weight_k = sample_weight[mask] if sample_weight is not None else None
            
            params = self.base_expert_params.copy()
            if k == 0:
                params.update({'max_depth': 8, 'reg_lambda': 1.0, 'learning_rate': 0.02, 'aft_loss_distribution': 'normal'})
            elif k == 1:
                params.update({'max_depth': 6, 'reg_lambda': 5.0, 'learning_rate': 0.015, 'aft_loss_distribution': 'logistic'})
            else:
                params.update({'max_depth': 4, 'reg_lambda': 15.0, 'learning_rate': 0.01, 'aft_loss_distribution': 'extreme'})

            X_tr, X_val, y_tr, y_val, c_tr, c_val, w_tr, w_val = train_test_split(
                X_k, y_k, censor_k, weight_k, test_size=0.15, random_state=42
            )

            dtrain = xgb.DMatrix(X_tr, weight=w_tr)
            dtrain.set_float_info('label_lower_bound', y_tr.values)
            dtrain.set_float_info('label_upper_bound', np.where(c_tr.values == 1, y_tr.values, np.inf))

            dval = xgb.DMatrix(X_val, weight=w_val)
            dval.set_float_info('label_lower_bound', y_val.values)
            dval.set_float_info('label_upper_bound', np.where(c_val.values == 1, y_val.values, np.inf))

            expert_model = xgb.train(
                params, dtrain, num_boost_round=600,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=25, verbose_eval=False
            )
            self.experts.append(expert_model)

        print("  -> Phase 2: Evaluating Expert Competence (Optimizing for MAPE)...")
        # Make all experts predict on the full dataset to find their exact error
        dtrain_full = xgb.DMatrix(X_train)
        expert_full_preds = np.zeros((X_train.shape[0], self.n_experts))
        
        for k, expert in enumerate(self.experts):
            expert_full_preds[:, k] = np.clip(expert.predict(dtrain_full), 1.0, 43200.0)
            
        # LITERATURE ALIGNMENT: Calculate Absolute Percentage Error (APE) instead of log-error
        # This explicitly forces the router to prioritize experts that minimize MAPE and MAE
        y_true_vals = y_train.values.reshape(-1, 1)
        ape_errors = np.abs((expert_full_preds - y_true_vals) / y_true_vals)
        
        # The target label for the router is the expert with the lowest Percentage Error
        best_expert_labels = np.argmin(ape_errors, axis=1)

        print("  -> Phase 3: Training MAPE-Optimized Router...")
        self.gating_network.fit(X_train, best_expert_labels, sample_weight=sample_weight)

    def predict(self, X_test):
        router_logits = self.gating_network.predict(X_test, output_margin=True)
        
        # Lowered temperature to 1.5 to trust the MAPE-optimized router more
        scaled_logits = router_logits / self.temperature
        gating_probs = softmax(scaled_logits, axis=1)

        expert_preds = np.zeros((X_test.shape[0], self.n_experts))
        dtest = xgb.DMatrix(X_test)
        
        for k, expert in enumerate(self.experts):
            raw_pred = expert.predict(dtest)
            expert_preds[:, k] = np.clip(raw_pred, 1.0, 43200.0)

        # Log-space geometric blending maintains the RMSE stabilization
        log_preds = np.log1p(expert_preds)
        final_log_predictions = np.sum(gating_probs * log_preds, axis=1)
        final_predictions = np.expm1(final_log_predictions)
        
        return final_predictions