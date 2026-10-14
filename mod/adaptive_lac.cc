#include "mod/adaptive_lac.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

namespace {
using adgMod::kMaxLacDegree;
using adgMod::kMinLacDegree;
using adgMod::IopsState;

std::atomic<int> g_history_window(adgMod::kDefaultHistoryWindow);

constexpr double kBaseUnitBytes = 2097152.0;  // 2MB
constexpr double kOpsEpsilon = 1e-6;
constexpr double kAlpha = 0.1;
constexpr double kGamma = 0.9;
constexpr double kEpsilonStart = 0.9;
constexpr double kEpsilonMin = 0.01;
constexpr double kEpsilonDecay = 0.008;
constexpr double kRangeEps = 1e-9;
constexpr double kPreferenceWeight = 0.4;
constexpr double kSmallDegreeBonus = 0.6;


//  The weights for each level are used to bias the reward function for the Q-learning algorithm. Higher levels (which typically have larger files and more compaction work) are given higher weights, encouraging the algorithm to prioritize optimizing compactions at those levels. The weights are defined in a constexpr array, allowing for compile-time evaluation and optimization.
constexpr double kLevelRewardWeights[leveldb::config::kNumLevels] = {
    0.6, 0.8, 0.9, 1.1, 1.3, 1.5, 1.7};

// The PreferredRange struct defines a range of preferred degrees for each level in the LAC algorithm. It contains two integer members, min_deg and max_deg, which represent the minimum and maximum preferred degrees, respectively. This struct is used to bias the selection of degrees during the Q-learning process, encouraging the algorithm to favor certain degree ranges that are considered optimal for specific levels.
struct PreferredRange {
  int min_deg;
  int max_deg;
};

// The kPreferredDegreeRanges array defines the preferred degree ranges for each level in the LAC algorithm. It is a constexpr array of PreferredRange structs, with one entry for each level (from L0 to L6). Each entry specifies the minimum and maximum preferred degrees for that level, allowing the Q-learning algorithm to bias its degree selection towards these ranges during the learning process.
std::string JoinPath(const std::string& base, const std::string& leaf) {
  if (base.empty()) return leaf;
  if (base.back() == '/') return base + leaf;
  return base + "/" + leaf;
}

// The ReportPath function determines the file path for the report.csv file used to log the results of the LAC algorithm. It checks for an environment variable "WT_LOG_DIR" to specify a custom log directory. If the environment variable is set and not empty, it joins that directory with "report.csv" to create the full path. If the environment variable is not set or empty, it defaults to "report.csv" in the current working directory. The function returns a reference to a static string containing the computed path, ensuring that the path is only computed once and reused on subsequent calls.
const std::string& ReportPath() {
  static const std::string path = [] {
    const char* log_dir = std::getenv("WT_LOG_DIR");
    if (log_dir != nullptr && log_dir[0] != '\0') {
      return JoinPath(log_dir, "report.csv");
    }
    return std::string("report.csv");
  }();
  return path;
}

// The kPreferredDegreeRanges array defines the preferred degree ranges for each level in the LAC algorithm. It is a constexpr array of PreferredRange structs, with one entry for each level (from L0 to L6). Each entry specifies the minimum and maximum preferred degrees for that level, allowing the Q-learning algorithm to bias its degree selection towards these ranges during the learning process.
constexpr std::array<PreferredRange, leveldb::config::kNumLevels>
    kPreferredDegreeRanges = {{
        {1, 10},  // L0: allow full range, bias handled separately
        {1, 10},  // L1
        {1, 10},   // L2
        {1, 10},   // L3
        {1, 10},   // L4
        {1, 10},   // L5
        {1, 10},   // L6
    }};

// The StateIndex function converts an IopsState enum value to its corresponding integer index. It uses static_cast to convert the IopsState value (which is of type uint8_t) to an int, allowing it to be used as an index in arrays or other data structures that require integer indices. This function is useful for accessing the appropriate row in the Q-values table based on the current IOPS state.
int StateIndex(adgMod::IopsState state) {
  return static_cast<int>(state);
}

// The ToString function converts an IopsState enum value to its corresponding string representation. It uses a switch statement to match the IopsState value to its string equivalent ("high", "medium", or "low"). If the state does not match any of the defined cases, it returns "unknown". This function is useful for logging or displaying the current IOPS state in a human-readable format.
const char* ToString(adgMod::IopsState state) {
  switch (state) {
    case adgMod::IopsState::kHigh:
      return "high";
    case adgMod::IopsState::kMedium:
      return "medium";
    case adgMod::IopsState::kLow:
      return "low";
  }
  return "unknown";
}

// The UpdateRange function updates the minimum and maximum values of a range based on a new input value. It takes a double value and references to min_val and max_val. If min_val is not finite or the new value is less than min_val, it updates min_val to the new value. Similarly, if max_val is not finite or the new value is greater than max_val, it updates max_val to the new value. This function is useful for tracking the range of observed values over time.
void UpdateRange(double value, double& min_val, double& max_val) {
  if (!std::isfinite(min_val) || value < min_val) {
    min_val = value;
  }
  if (!std::isfinite(max_val) || value > max_val) {
    max_val = value;
  }
}

// The Normalize function normalizes a given value within a specified range defined by min_val and max_val. It checks if the min_val and max_val are finite and if the range (max_val - min_val) is greater than a small epsilon (kRangeEps). If the conditions are not met, it returns 0.5 as a default normalized value. Otherwise, it scales the input value to a range between 0.0 and 1.0 based on its position within the specified range, ensuring that the result is clamped between 0.0 and 1.0. This function is useful for converting raw measurements into a standardized format for further processing or analysis.
double Normalize(double value, double min_val, double max_val) {
  if (!std::isfinite(min_val) || !std::isfinite(max_val) ||
      (max_val - min_val) < kRangeEps) {
    return 0.5;
  }
  const double scaled = (value - min_val) / (max_val - min_val);
  return std::max(0.0, std::min(1.0, scaled));
}

// The ClassifyIops function classifies the normalized IOPS (Input/Output Operations Per Second) value into one of three states: high, medium, or low. It takes a double value representing the normalized IOPS and compares it against predefined thresholds. If the normalized IOPS is greater than or equal to 0.66, it returns the kHigh state. If it is greater than or equal to 0.33 but less than 0.66, it returns the kMedium state. Otherwise, it returns the kLow state. This classification helps in determining the current performance state of the system based on IOPS measurements.
adgMod::IopsState ClassifyIops(double normalized_ops) {
  if (normalized_ops >= 0.66) {
    return adgMod::IopsState::kHigh;
  }
  if (normalized_ops >= 0.33) {
    return adgMod::IopsState::kMedium;
  }
  return adgMod::IopsState::kLow;
}

// The SmallDegreeBias function calculates a bias value for small degrees based on the current level and degree. It is designed to favor smaller degrees (<=3) for lower levels (0 and 1) while still allowing the full range of degrees to be valid. If the level is 0 or 1 and the degree is less than or equal to 3, it returns a positive bias (kSmallDegreeBonus). For degrees greater than 3, it calculates a negative bias that scales linearly with the degree, ensuring that the bias decreases as the degree increases. For levels greater than 1, it returns a bias of 0. This function helps guide the Q-learning algorithm towards preferred degree ranges based on the current level.
double SmallDegreeBias(int level, int degree) {
  // Bias levels 0/1 toward very small degrees (<=3), but keep full range valid.
  if (level <= 1) {
    if (degree <= 3) return kSmallDegreeBonus;
    const double scale = static_cast<double>(degree - 3) /
                         static_cast<double>(kMaxLacDegree - 3);
    return -kSmallDegreeBonus * std::min(1.0, std::max(0.0, scale));
  }
  return 0.0;
}

// The GetRng function provides a thread-local random number generator (RNG) using the Mersenne Twister algorithm (std::mt19937). It initializes the RNG with a seed from std::random_device, ensuring that each thread has its own independent RNG instance. This is important for generating random numbers in a multi-threaded environment without causing contention or sharing state between threads. The function returns a reference to the thread-local RNG, allowing it to be used for generating random values in various parts of the code.
std::mt19937& GetRng() {
  static thread_local std::mt19937 rng(std::random_device{}());
  return rng;
}

// The NextEpsilonGreedy function implements an epsilon-greedy strategy for exploration in reinforcement learning. It takes a reference to the current epsilon value, which represents the probability of choosing a random action (exploration) versus the best-known action (exploitation). The function returns the current epsilon value and then decays it by a fixed amount (kEpsilonDecay), ensuring that it does not go below a minimum threshold (kEpsilonMin). This allows the algorithm to gradually shift from exploration to exploitation as it learns more about the environment.
double NextEpsilonGreedy(double& epsilon) {
  const double current = epsilon;
  epsilon = std::max(kEpsilonMin, epsilon - kEpsilonDecay);
  return current;
}

// The RandomDegree function generates a random degree value within a specified range (min_degree to max_degree) using a uniform integer distribution. It creates a std::uniform_int_distribution object with the given range and uses the thread-local random number generator (GetRng) to produce a random integer. This function is useful for selecting a random degree for the LAC algorithm during exploration phases of the Q-learning process.
int RandomDegree(int min_degree, int max_degree) {
  std::uniform_int_distribution<int> dist(min_degree, max_degree);
  return dist(GetRng());
}

// The ArgMaxDegree function finds the degree with the maximum Q-value within a specified range (min_degree to max_degree). It takes an array of Q-values and iterates through the specified range, keeping track of the best Q-value and its corresponding degree. If a higher Q-value is found, it updates the best value and degree. The function returns the degree that has the highest Q-value, which is used for exploitation in the Q-learning process.
int ArgMaxDegree(const std::array<double, kMaxLacDegree + 1>& q_values,
                 int min_degree, int max_degree) {
  double best = q_values[min_degree];
  int best_deg = min_degree;
  for (int d = min_degree + 1; d <= max_degree; ++d) {
    if (q_values[d] > best) {
      best = q_values[d];
      best_deg = d;
    }
  }
  return best_deg;
}

// The MaxQValue function finds the maximum Q-value within a specified range of degrees (min_degree to max_degree) from an array of Q-values. It initializes the best value with the Q-value at min_degree and iterates through the range, updating the best value whenever a higher Q-value is found. The function returns the highest Q-value found in the specified range, which is used in the Q-learning update rule to estimate the expected future rewards for a given state-action pair.
double MaxQValue(const std::array<double, kMaxLacDegree + 1>& q_values,
                 int min_degree, int max_degree) {
  double best = q_values[min_degree];
  for (int d = min_degree + 1; d <= max_degree; ++d) {
    best = std::max(best, q_values[d]);
  }
  return best;
}

// The SelectNextDegree function selects the next degree for the LAC algorithm based on the current Q-values and an epsilon-greedy strategy. It takes an array of Q-values, a minimum and maximum degree, and a reference to the current epsilon value. The function generates a random number and compares it to the current epsilon value (obtained from NextEpsilonGreedy). If the random number is less than the current epsilon, it selects a random degree within the specified range (exploration). Otherwise, it selects the degree with the highest Q-value (exploitation) using ArgMaxDegree. The selected degree is returned as the next degree to be used in the LAC algorithm.
int SelectNextDegree(const std::array<double, kMaxLacDegree + 1>& q_values,
                     int min_degree, int max_degree, double& epsilon) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const double current_eps = NextEpsilonGreedy(epsilon);
  if (dist(GetRng()) < current_eps) {
    return RandomDegree(min_degree, max_degree);
  }
  return ArgMaxDegree(q_values, min_degree, max_degree);
}

// The TrimDeque function trims a deque of double values to ensure that its size does not exceed a specified window size. It takes a reference to a std::deque<double> and a reference to a double sum. The function calculates the maximum allowed window size using GetAdaptiveLacWindowSize() and removes elements from the front of the deque while its size exceeds the window size. As elements are removed, their values are subtracted from the sum, maintaining an accurate total of the remaining elements in the deque. This function is useful for maintaining a sliding window of recent observations for statistical calculations in the LAC algorithm.
void TrimDeque(std::deque<double>& dq, double& sum) {
  const int window = std::max(1, adgMod::GetAdaptiveLacWindowSize());
  while (dq.size() > static_cast<size_t>(window)) {
    sum -= dq.front();
    dq.pop_front();
  }
}
// The AppendReportRow function appends a row of data to a report CSV file, logging various metrics related to the LAC algorithm's performance. It takes parameters such as the level, IOPS states before and after the observation, old and new degrees, write amplification (wa), operations (ops), their respective rates, score, level weight, and target bytes. The function uses a mutex to ensure thread safety when writing to the file. It initializes the report file with headers if it hasn't been done yet and formats the output with fixed precision. The current timestamp is also included in the report row. This function is useful for tracking the performance and decisions made by the LAC algorithm over time for analysis and debugging purposes.
void AppendReportRow(int level, IopsState from_state, IopsState to_state,
                     int old_deg, int new_deg, double wa, double ops,
                     double wa_rate, double ops_rate, double score,
                     double level_weight, double target_bytes) {
  static std::mutex report_mu;
  static bool report_initialized = false;
  std::lock_guard<std::mutex> lg(report_mu);
  
  
  std::ios_base::openmode mode = std::ios::app;
  if (!report_initialized) {
    mode = std::ios::out | std::ios::trunc;
  }
  std::ofstream ofs(ReportPath(), mode);
  if (!ofs.is_open()) {
    return;
  }
  if (!report_initialized) {
    ofs << "ts_iso,level,old_lac,new_lac,wa,ops,wa_rate,ops_rate,score,"
           "level_weight,target_bytes,iops_from,iops_to\n";
    report_initialized = true;
  }

  const auto now = std::chrono::system_clock::now();
  const std::time_t ts_sec = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf;
#if defined(_WIN32)
  localtime_s(&tm_buf, &ts_sec);
#else
  localtime_r(&ts_sec, &tm_buf);
#endif
  std::ostringstream ts_stream;
  ts_stream << std::put_time(&tm_buf, "%F %T");

  const auto old_flags = ofs.flags();
  const auto old_prec = ofs.precision();
  ofs << std::fixed << std::setprecision(6);

  ofs << ts_stream.str() << ',' << level << ',' << old_deg << ',' << new_deg
      << ',' << wa << ',' << ops << ',' << wa_rate << ',' << ops_rate << ','
      << score << ',' << level_weight << ','
      << static_cast<uint64_t>(target_bytes) << ',' << ToString(from_state)
      << ',' << ToString(to_state) << '\n';

  ofs.flags(old_flags);
  ofs.precision(old_prec);
}
}  // namespace

namespace adgMod {

  // The GetAdaptiveLacWindowSize function retrieves the current window size for the Adaptive LAC algorithm. It uses an atomic integer (g_history_window) to store the window size, ensuring thread-safe access. The function returns the maximum of 1 and the current value of g_history_window, which is loaded with relaxed memory order. This ensures that the window size is always at least 1, preventing issues with empty or invalid windows during statistical calculations in the LAC algorithm.
int GetAdaptiveLacWindowSize() {
  return std::max(1, g_history_window.load(std::memory_order_relaxed));
}

// The SetAdaptiveLacWindowSize function sets the window size for the Adaptive LAC algorithm. It takes an integer parameter (window) representing the desired window size. The function clamps the input value to ensure it is at least 1, preventing invalid or empty windows. It then stores the clamped value in the atomic integer g_history_window using relaxed memory order, allowing for thread-safe updates to the window size. This function allows users to configure the history window size used in statistical calculations for the LAC algorithm.
void SetAdaptiveLacWindowSize(int window) {
  const int clamped = std::max(1, window);
  g_history_window.store(clamped, std::memory_order_relaxed);
}

// The GetAdaptiveLac function provides access to a singleton instance of the AdaptiveLAC class. It uses a static local variable to ensure that only one instance of AdaptiveLAC is created and shared across the application. The function returns a reference to this instance, allowing other parts of the code to interact with the AdaptiveLAC functionality without needing to create multiple instances. This design pattern is useful for managing shared state and resources in a controlled manner.
AdaptiveLAC::AdaptiveLAC() {
  for (size_t level = 0; level < states_.size(); ++level) {
    State& st = states_[level];
    st.target_bytes = 0.0;
    st.last_observed_bytes = 0;
    st.last_ops_total = 0;
    st.ops_min = std::numeric_limits<double>::infinity();
    st.ops_max = -std::numeric_limits<double>::infinity();
    st.iops_state = IopsState::kMedium;
    st.epsilon = kEpsilonStart;
    st.lac_degree =
        ClampDegreeForLevel(static_cast<int>(level),
                            InitialLacDegree(static_cast<int>(level)));
    st.initialized = false;
    for (auto& q_row : st.q_values) {
      q_row.fill(0.0);
    }
    for (auto& hist : st.degree_stats) {
      hist = DegreeStats();
    }
  }
}

// The BaseSizeForLevel function calculates the base size in bytes for a given level and LAC degree. It takes an integer level and an integer lac_degree as parameters. The function first determines the effective degree to use, which is either the provided lac_degree (if greater than 0) or the initial LAC degree for the specified level. It then calculates an exponent based on the level, ensuring it is between 1 and 6. The base size is computed using a formula that multiplies a constant base unit (kBaseUnitBytes) by the degree raised to the power of the exponent. Finally, the function returns the calculated size as a uint64_t value, representing the base size in bytes for the specified level and degree.
uint64_t AdaptiveLAC::BaseSizeForLevel(int level, int lac_degree) const {
  const int degree = lac_degree > 0 ? lac_degree : InitialLacDegree(level);
  const int exp = std::max(1, std::min(level + 1, 6));
  const double size = kBaseUnitBytes *
                      std::pow(static_cast<double>(degree),
                               static_cast<double>(exp));
  return static_cast<uint64_t>(size);
}

// The MinDegreeForLevel function returns the minimum allowed LAC degree for a given level. It takes an integer level as a parameter and clamps it to ensure it is within the valid range of levels (0 to kNumLevels - 1). The function retrieves the corresponding minimum degree from the kMinDegreePerLevel array and ensures that it is at least kMinLacDegree. This function is used to enforce constraints on the LAC degree based on the level, ensuring that the degree remains within acceptable bounds for the algorithm's operation.
int AdaptiveLAC::MinDegreeForLevel(int level) {
  const int idx =
      std::max(0, std::min(level, leveldb::config::kNumLevels - 1));
  return std::max(kMinLacDegree, adgMod::kMinDegreePerLevel[idx]);
}

// The MaxDegreeForLevel function returns the maximum allowed LAC degree for a given level. It takes an integer level as a parameter and simply returns the constant kMaxLacDegree, which represents the upper limit for the LAC degree across all levels. This function is used to enforce constraints on the LAC degree, ensuring that it does not exceed the defined maximum value during the algorithm's operation.
int AdaptiveLAC::MaxDegreeForLevel(int level) {
  return kMaxLacDegree;
}

// The RewardWeightForLevel function returns the reward weight for a given level in the LAC algorithm. It takes an integer level as a parameter and checks if the level is within the valid range (0 to kNumLevels - 1). If the level is out of bounds, it returns a default weight of 1.0. Otherwise, it retrieves the corresponding weight from the kLevelRewardWeights array, which defines specific weights for each level to bias the reward function in the Q-learning algorithm. This function is used to adjust the influence of rewards based on the level being processed, allowing for more effective learning and optimization of compaction strategies.
double RewardWeightForLevel(int level) {
  if (level < 0 || level >= leveldb::config::kNumLevels) return 1.0;
  return kLevelRewardWeights[level];
}

// The ClampDegreeForLevel function ensures that a given LAC degree is within the valid range for a specified level. It takes an integer level and an integer degree as parameters. The function retrieves the minimum and maximum allowed degrees for the specified level using MinDegreeForLevel and MaxDegreeForLevel, respectively. It then clamps the input degree to be no less than the minimum and no greater than the maximum, returning the adjusted degree. This function is used to enforce constraints on the LAC degree based on the level, ensuring that it remains within acceptable bounds during the algorithm's operation.
int AdaptiveLAC::ClampDegreeForLevel(int level, int degree) {
  const int min_degree = MinDegreeForLevel(level);
  const int max_degree = MaxDegreeForLevel(level);
  return std::max(min_degree, std::min(max_degree, degree));
}

// The InitialLacDegree function determines the initial LAC degree for a given level. It takes an integer level as a parameter and calculates a base degree value based on the global variable sst_size. If sst_size is greater than 0, it uses that value; otherwise, it defaults to 4. The function then calls ClampDegreeForLevel to ensure that the calculated base degree is within the valid range for the specified level, returning the clamped degree. This function is used to establish a starting point for the LAC degree when initializing the algorithm for a specific level.
int AdaptiveLAC::InitialLacDegree(int level) const {
  const int base = sst_size > 0 ? sst_size : 4;
  return ClampDegreeForLevel(level, base);
}

uint64_t AdaptiveLAC::Clamp(uint64_t value, uint64_t base) {
  const double min_bytes = base * 0.5;
  const double max_bytes = base * 4.0;
  return static_cast<uint64_t>(
      std::max(min_bytes, std::min<double>(value, max_bytes)));
}

uint64_t AdaptiveLAC::DefaultSizeForLevel(int level) const {
  return BaseSizeForLevel(level, InitialLacDegree(level));
}

uint64_t AdaptiveLAC::GetTargetBytes(int level, uint64_t fallback_bytes) const {
  std::lock_guard<std::mutex> lg(mu_);
  if (level < 0 || level >= static_cast<int>(states_.size())) {
    return fallback_bytes ? fallback_bytes
                          : BaseSizeForLevel(0, InitialLacDegree(0));
  }

  const State& st = states_[level];
  const int lac_degree =
      st.initialized ? ClampDegreeForLevel(level, st.lac_degree)
                     : InitialLacDegree(level);
  const uint64_t base = fallback_bytes ? fallback_bytes
                                       : BaseSizeForLevel(level, lac_degree);
  const double target = st.initialized ? st.target_bytes
                                       : static_cast<double>(base);
  return Clamp(static_cast<uint64_t>(target), base);
}

void AdaptiveLAC::Observe(int level, uint64_t file_size_bytes, double wa_score, uint64_t compaction_micros, double ops) {
  (void)compaction_micros;
  std::lock_guard<std::mutex> lg(mu_);
  if (level < 0 || level >= static_cast<int>(states_.size())) {
    return;
  }

  if (wa_score <= 0.0) {
    wa_score = 1.0;
  }

  State& st = states_[level];

  if (!st.initialized) {
    st.lac_degree = InitialLacDegree(level);
    const uint64_t base = BaseSizeForLevel(level, st.lac_degree);
    st.target_bytes = file_size_bytes ? static_cast<double>(file_size_bytes)
                                      : static_cast<double>(base);
    st.last_observed_bytes = file_size_bytes;
    st.epsilon = kEpsilonStart;
    st.initialized = true;
  }

  const int min_degree = MinDegreeForLevel(level);
  const int max_degree = MaxDegreeForLevel(level);
  const int current_degree = ClampDegreeForLevel(level, st.lac_degree);
  DegreeStats& stats = st.degree_stats[current_degree];
  const IopsState current_state = st.iops_state;

  const uint64_t ops_total = static_cast<uint64_t>(std::max(0.0, ops));
  double ops_measure = 0.0;
  if (st.last_ops_total > 0 && ops_total >= st.last_ops_total) {
    ops_measure = static_cast<double>(ops_total - st.last_ops_total);
  } else if (st.last_ops_total == 0) {
    ops_measure = static_cast<double>(ops_total);
  } else {
    ops_measure = 0.0;
  }
  st.last_ops_total = ops_total;

  if (stats.last_ops_measure > 0.0) {
    const double rel = (ops_measure - stats.last_ops_measure) /
                       std::max(stats.last_ops_measure, kOpsEpsilon);
    stats.ops_changes.push_back(rel);
    stats.sum_ops += rel;
    TrimDeque(stats.ops_changes, stats.sum_ops);
  }
  stats.last_ops_measure = ops_measure;

  if (stats.last_wa > 0.0) {
    const double rel = (wa_score - stats.last_wa) /
                       std::max(stats.last_wa, kOpsEpsilon);
    stats.wa_changes.push_back(rel);
    stats.sum_wa += rel;
    TrimDeque(stats.wa_changes, stats.sum_wa);
  }
  stats.last_wa = wa_score;

  const double r_ops = stats.ops_changes.empty()
                           ? 0.0
                           : stats.sum_ops / stats.ops_changes.size();
  const double r_wa = stats.wa_changes.empty()
                          ? 0.0
                          : stats.sum_wa / stats.wa_changes.size();
  UpdateRange(r_ops, stats.min_ops_change, stats.max_ops_change);
  UpdateRange(r_wa, stats.min_wa_change, stats.max_wa_change);
  const double norm_ops =
      Normalize(r_ops, stats.min_ops_change, stats.max_ops_change);
  const double norm_wa =
      Normalize(r_wa, stats.min_wa_change, stats.max_wa_change);
  const double level_weight = RewardWeightForLevel(level);
  double reward = (norm_ops - norm_wa) * level_weight;

  // Add a small bias toward preferred degree ranges per level (e.g., 2/3 at
  // lower levels, 3–5 at mid levels).
  const int level_idx =
      std::max(0, std::min(level, leveldb::config::kNumLevels - 1));
  const PreferredRange& pref = kPreferredDegreeRanges[level_idx];
  const bool in_pref =
      current_degree >= pref.min_deg && current_degree <= pref.max_deg;
  const double range_bias = (in_pref ? kPreferenceWeight : -kPreferenceWeight);
  const double small_bias = SmallDegreeBias(level, current_degree);
  reward += (range_bias + small_bias) * level_weight;
  if (level == 2) {
    // Extra soft preference for degree 4 while keeping full 1-10 range valid.
    const double dist = std::abs(current_degree - 4);
    const double l2_bias = std::max(0.0, 1.0 - dist / 6.0);  // 4 gets full bonus, farther ones decay.
    const double l2_weight = 0.5;  // tune bias strength
    reward += l2_weight * l2_bias * level_weight;
  }

  UpdateRange(ops_measure, st.ops_min, st.ops_max);
  const double normalized_iops = Normalize(ops_measure, st.ops_min, st.ops_max);
  const IopsState next_state = ClassifyIops(normalized_iops);

  auto& current_q_row = st.q_values[StateIndex(current_state)];
  auto& next_q_row = st.q_values[StateIndex(next_state)];
  double& q = current_q_row[current_degree];
  const double max_next =
      MaxQValue(next_q_row, min_degree, max_degree);
  q += kAlpha * (reward + kGamma * max_next - q);

  const int next_degree =
      SelectNextDegree(next_q_row, min_degree, max_degree, st.epsilon);
  st.iops_state = next_state;
  st.lac_degree = next_degree;

  const uint64_t base = BaseSizeForLevel(level, st.lac_degree);
  double next_target = static_cast<double>(base);
  if (file_size_bytes > 0) {
    next_target = 0.5 * next_target + 0.5 * static_cast<double>(file_size_bytes);
  }
  st.target_bytes = static_cast<double>(Clamp(
      static_cast<uint64_t>(next_target), base));
  st.last_observed_bytes = file_size_bytes;

  AppendReportRow(level, current_state, next_state, current_degree, next_degree,
                  wa_score, ops_measure, r_wa, r_ops, reward, level_weight,
                  st.target_bytes);
}

AdaptiveLAC& GetAdaptiveLAC() {
  static AdaptiveLAC instance;
  return instance;
}

}  // namespace adgMod
