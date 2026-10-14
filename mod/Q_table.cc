#include "Q_table.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

template <typename T>
inline T clamp(const T& val, const T& low, const T& high)
{
    if (val < low) {
        return low;
    }
    if (val > high) {
        return high;
    }
    return val;
}

namespace adgMod {
    // 返回固定的 error_bound 动作集合
    const std::vector<double>& ErrorBoundActions() {
        static const std::vector<double> actions =
            {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64};
        return actions;
    }
    // this function returns a filtered list of error_bound actions based on the block size.
    std::vector<double> ErrorBoundActionsForBlockSize(uint64_t block_size) {
        std::vector<double> actions;
        const double max_action = (block_size > 0 && block_size <= 4096) ? 44.0 : 64.0;
        for (double action : ErrorBoundActions()) {
            if (action <= max_action) {
                actions.push_back(action);
            }
        }
        return actions.empty() ? ErrorBoundActions() : actions;
    }

    // QTableManager class implementation
    QTableManager::QTableManager() {
        error_bound_actions = ErrorBoundActionsForBlockSize(0);
        latency_ema.fill(0.0);
        latency_min.fill(std::numeric_limits<double>::infinity());
        latency_max.fill(-std::numeric_limits<double>::infinity());
        latency_seen.fill(false);
    }

    double QTableManager::getLearningRate(int state, double action)
    {
        int n = Q_table[state].visit_counts[action];
        return 1.0 / std::max(1, n);
    }

    double QTableManager::compute_reward(
        double   action,
        double   load_cost,
        double   pred_cost,
        double   corr_cost,
        uint64_t key_count
    )
    {
        (void)action;
        CostBounds& bounds = cost_bounds;
        if (bounds.min_load == std::numeric_limits<double>::max()) {
            bounds.min_load = bounds.max_load = load_cost;
            bounds.min_pred = bounds.max_pred = pred_cost;
            bounds.min_corr = bounds.max_corr = corr_cost;
        }
        bounds.min_load = std::min(bounds.min_load, load_cost);
        bounds.max_load = std::max(bounds.max_load, load_cost);
        bounds.min_pred = std::min(bounds.min_pred, pred_cost);
        bounds.max_pred = std::max(bounds.max_pred, pred_cost);
        bounds.min_corr = std::min(bounds.min_corr, corr_cost);
        bounds.max_corr = std::max(bounds.max_corr, corr_cost);
        bounds.max_keys = std::max(bounds.max_keys, key_count);

        auto normalize = [](double value, double min_value, double max_value) {
            const double delta = max_value - min_value;
            if (delta <= 1e-12) return 0.0;
            return clamp((value - min_value) / delta, 0.0, 1.0);
        };

        const double norm_load = normalize(load_cost, bounds.min_load, bounds.max_load);
        const double norm_pred = normalize(pred_cost, bounds.min_pred, bounds.max_pred);
        const double norm_corr = normalize(corr_cost, bounds.min_corr, bounds.max_corr);
        const double r_cost = 1.0 - (norm_load + norm_pred + norm_corr) / 3.0;

        double size_penalty = 0.0;
        if (bounds.max_keys > 0 && key_count > 0) {
            size_penalty = 1.0 - std::log(static_cast<double>(key_count) + 1.0) /
                                     std::log(static_cast<double>(bounds.max_keys) + 1.0);
            size_penalty = clamp(size_penalty, 0.0, 1.0);
        }
        return r_cost - size_penalty;
    }

    double QTableManager::get_max_future_q(int next_state) const {
        double max_q = -std::numeric_limits<double>::infinity();
        for (const auto& pair : Q_table[next_state].q_values) {
            if (pair.second > max_q) {
                max_q = pair.second;
            }
        }
        return (max_q == -std::numeric_limits<double>::infinity()) ? 0.0 : max_q;
    }

    double QTableManager::get_max_future_q(int next_state, const std::vector<double>& actions) const {
        double max_q = -std::numeric_limits<double>::infinity();
        for (const auto& action : actions) {
            auto it = Q_table[next_state].q_values.find(action);
            if (it != Q_table[next_state].q_values.end()) {
                if (it->second > max_q) {
                    max_q = it->second;
                }
            }
        }
        return (max_q == -std::numeric_limits<double>::infinity()) ? 0.0 : max_q;
    }

    double QTableManager::compute_q_value(int state,
                                      double action_raw,
                                      double reward,
                                      int    next_state)
    {
        const double action = std::round(action_raw * 1e4) / 1e4;

        const double alpha  = getLearningRate(state, action);
        const double gamma  = std::pow(2.0, -1.0 / 3.0);

        double& prev_q = Q_table[state].q_values[action];

        auto& next_map = Q_table[next_state].q_values;
        double max_future_q = 0.0;
        if (next_map.empty()) {
            max_future_q = 1.0;
        } else {
            for (const auto& kv : next_map)
                if (kv.second > max_future_q) max_future_q = kv.second;
        }

        const double td_target = reward + gamma * max_future_q;
        double td_error        = td_target - prev_q;

        const double delta_clip = 2.0;
        if (td_error >  delta_clip) td_error =  delta_clip;
        if (td_error < -delta_clip) td_error = -delta_clip;

        prev_q += alpha * td_error;
        return prev_q;
    }

    double QTableManager::compute_q_value(int state, double action, double reward, int next_state, double next_action) {
        double alpha = getLearningRate(state, action);
        double gamma = 0.8;

        double& prev_q = Q_table[state].q_values[action];
        double next_q = Q_table[next_state].q_values[next_action];
        double Q_value = (1 - alpha) * prev_q + alpha * (reward + gamma * next_q);
        prev_q = Q_value;
        return Q_value;
    }

    void QTableManager::updateQValue(int state, double action, double Q_value) {
        Q_table[state].q_values[action] = Q_value;
        Q_table[state].visit_counts[action] += 1;
    }

    double QTableManager::getErrorBound(int state) const {
        return 8;
    }

    void QTableManager::addExperience(int state, double action, double reward, int next_state) {
        Experience exp = {state, action, reward, next_state};
        replay_buffer.push_back(exp);
        if (replay_buffer.size() > max_replay_size) {
            replay_buffer.erase(replay_buffer.begin());
        }
    }

    void QTableManager::learnFromReplay() {
        if (replay_buffer.size() < batch_size) return;

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, replay_buffer.size() - 1);

        for (size_t i = 0; i < batch_size; ++i) {
            int idx = dis(gen);
            Experience& exp = replay_buffer[idx];
            double Q_value;
            Q_value = compute_q_value(exp.state, exp.action, exp.reward, exp.next_state);

            updateQValue(exp.state, exp.action, Q_value);
        }
    }

    void QTableManager::initQTable(uint64_t block_size) {
        Q_table.clear();
        Q_table.resize(kErrorQStates);
        error_bound_actions = ErrorBoundActionsForBlockSize(block_size);
        cost_bounds = CostBounds();
        id_history.clear();
        id_set.clear();
        id_bounds_fixed = false;
        id_observations = 0;
        latency_ema.fill(0.0);
        latency_min.fill(std::numeric_limits<double>::infinity());
        latency_max.fill(-std::numeric_limits<double>::infinity());
        latency_seen.fill(false);
        for (auto& entry : Q_table) {
            for (double action : error_bound_actions) {
                entry.q_values[action] = 0.0;
                entry.visit_counts[action] = 0;
                entry.last_action = 16.0;
            }
        }
    }

    int QTableManager::ObserveID(double inverse_density) {
        onNewSSTableID(inverse_density);
        if (ID_max <= ID_min) return 0;
        if (inverse_density < ID_1) return 0;
        if (inverse_density < ID_2) return 1;
        if (inverse_density < ID_3) return 2;
        return 3;
    }

    int QTableManager::ComposeState(int id_bucket, int latency_bucket) const {
        id_bucket = std::max(0, std::min(kErrorIdBuckets - 1, id_bucket));
        latency_bucket = std::max(0, std::min(kErrorLatencyBuckets - 1, latency_bucket));
        return id_bucket * kErrorLatencyBuckets + latency_bucket;
    }

    int QTableManager::LatencyBucket(int id_bucket) const {
        id_bucket = std::max(0, std::min(kErrorIdBuckets - 1, id_bucket));
        if (!latency_seen[id_bucket]) return 1;
        const double min_value = latency_min[id_bucket];
        const double max_value = latency_max[id_bucket];
        if (max_value <= min_value) return 1;
        const double normalized =
            clamp((latency_ema[id_bucket] - min_value) / (max_value - min_value), 0.0, 1.0);
        if (normalized < 0.33) return 0;
        if (normalized < 0.66) return 1;
        return 2;
    }

    int QTableManager::ObserveLatencyAndGetState(int id_bucket, double lookup_cost) {
        id_bucket = std::max(0, std::min(kErrorIdBuckets - 1, id_bucket));
        if (!latency_seen[id_bucket]) {
            latency_ema[id_bucket] = lookup_cost;
            latency_min[id_bucket] = lookup_cost;
            latency_max[id_bucket] = lookup_cost;
            latency_seen[id_bucket] = true;
        } else {
            latency_ema[id_bucket] = 0.8 * latency_ema[id_bucket] + 0.2 * lookup_cost;
            latency_min[id_bucket] = std::min(latency_min[id_bucket], latency_ema[id_bucket]);
            latency_max[id_bucket] = std::max(latency_max[id_bucket], latency_ema[id_bucket]);
        }
        return ComposeState(id_bucket, LatencyBucket(id_bucket));
    }

    QTableManager& getQTableManagerInstance() {
        static QTableManager instance;
        return instance;
    }

}
