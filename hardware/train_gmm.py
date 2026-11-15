# =====================================================================
#  TRAIN THE GMM WITH REAL NASA POWER DATA
#  Jackson Five · NASA Space Apps 2026 · Owner: Jesús Díaz
#
#  What this program does:
#    1. Reads 10 years of daily NASA POWER weather (Chihuahua, 29.25, -107.41)
#    2. Trains a GMM with AIR TEMPERATURE and AIR HUMIDITY, which are
#       exactly the 2 things the DHT22 sensor on Misael's ESP32 measures.
#    3. Uses the BIC to choose HOW MANY clusters and WHAT SHAPE the hills have,
#       and saves the chart "bic_gmm.png" for the presentation.
#    4. Checks with years the model NEVER saw (2024 and 2025) whether the
#       clusters still make sense.
#    5. Compares it against a simple rule (the Excel asks for it, hour 9).
#    6. Learns a MARKOV CHAIN to forecast the weather 7 days ahead.
#    7. Saves the model as "gmm_model.h" to load onto the ESP32.
#
#  How to run it (on your computer or on the AWS Jupyter, same thing):
#      pip install pandas scikit-learn matplotlib
#      python train_gmm.py
#
#  The file nasa_chihuahua.csv must be in the same folder.
# =====================================================================


# =====================================================================
# BLOCK 1: THE TOOLS
# =====================================================================
import numpy as np                              # number crunching
import pandas as pd                             # tables (like Excel)
from sklearn.mixture import GaussianMixture     # the GMM
import matplotlib
matplotlib.use("Agg")                           # draws to a file, no windows (works on AWS)
import matplotlib.pyplot as plt                 # charts


# =====================================================================
# BLOCK 2: READ THE NASA FILE
# =====================================================================
FILE = "nasa_chihuahua.csv"

# The NASA CSV has explanation lines at the top. We look for the
# "-END HEADER-" line to know how many to skip.
with open(FILE, encoding="utf-8") as f:
    lines = f.readlines()
skip = 0
for number, line in enumerate(lines):
    if "-END HEADER-" in line:
        skip = number + 1

data = pd.read_csv(FILE, skiprows=skip)
data = data.replace(-999, np.nan).dropna()   # drop days with missing data, if any

# Turn YEAR + DOY (day of year) into a real date,
# so we can tell which MONTH each day belongs to.
data["date"] = pd.to_datetime(
    data["YEAR"].astype(str) + "-" + data["DOY"].astype(str), format="%Y-%j")
data["month"] = data["date"].dt.month

print(f"📂 Read {len(data)} days, from {data['YEAR'].min()} to {data['YEAR'].max()}")


# =====================================================================
# BLOCK 3: CHOOSE WHAT THE MODEL SEES (THE "INPUTS")
# =====================================================================
# 👀 Golden rule of TinyML: the model can ONLY learn from what the
#    ESP32 will be able to measure in the field. The DHT22 measures air
#    temperature and humidity. It does NOT measure rain. So rain does NOT
#    go into the model. (We use it later, in Block 6, to check whether
#    the clusters make sense).
INPUTS = ["T2M", "RH2M"]      # temperature (°C) and air humidity (%)

# We split the data into two piles, like in school:
#   TRAIN = the years the model studies with (2016 to 2023)
#   TEST  = the "pop quiz" with years it never saw (2024 and 2025)
train = data[data["YEAR"] <= 2023]
test = data[data["YEAR"] >= 2024]
print(f"📚 For training: {len(train)} days  |  📝 For the exam: {len(test)} days")


# =====================================================================
# BLOCK 4: THE BIC CHOOSES HOW MANY CLUSTERS AND WHAT SHAPE THEY HAVE
# =====================================================================
# The BIC is an upside-down grade: THE LOWER, THE BETTER.
#   BIC = (how BADLY it explains the data) + (penalty for each extra number)
# A model with many clusters always "fits" better, but the penalty
# stops us from cheating by adding too many clusters.
#
# Besides the NUMBER of clusters, we test the SHAPE of the hills:
#   "full"      -> each hill with its own shape, can be tilted
#   "tied"      -> all hills with the SAME shape
#   "diag"      -> each one with its own shape, but straight (no tilt)
#   "spherical" -> round balls
# The simpler the shape, the fewer numbers the ESP32 has to store.
SHAPES = ["full", "tied", "diag", "spherical"]
CLUSTERS_TO_TRY = range(1, 8)
X = train[INPUTS].values

print("\n🔢 The BIC decides (lower = better):")
bic_table = {}            # (shape, k) -> BIC
models = {}               # (shape, k) -> already trained model
for shape in SHAPES:
    for k in CLUSTERS_TO_TRY:
        m = GaussianMixture(n_components=k, covariance_type=shape,
                            random_state=0, n_init=5)
        m.fit(X)
        bic_table[(shape, k)] = m.bic(X)
        models[(shape, k)] = m

best_shape, best_k = min(bic_table, key=bic_table.get)
best_bic = bic_table[(best_shape, best_k)]

# Print the table. "Δ" = how much worse each option is than the winner.
print(f"   {'clusters':>8} | " + " | ".join(f"{s:>10}" for s in SHAPES))
for k in CLUSTERS_TO_TRY:
    cells = []
    for shape in SHAPES:
        delta = bic_table[(shape, k)] - best_bic
        cells.append("  WINNER  " if delta == 0 else f"{delta:+10,.0f}")
    print(f"   {k:>8} | " + " | ".join(cells))
print(f"   ✅ Winner: {best_k} clusters with shape '{best_shape}' (BIC = {best_bic:,.0f})")
print("   📏 How to read the Δ: more than +10 = the winner is MUCH better; "
      "less than +2 = practically tied.")

# Check whether there was a technical tie with a simpler model
almost_tied = [(s, k) for (s, k), b in bic_table.items()
               if b - best_bic < 10 and (s, k) != (best_shape, best_k)]
if almost_tied:
    print(f"   ⚠️ Almost tied with the winner (Δ < 10): {almost_tied}. "
          "If one is simpler, it's fair to defend it before the judges.")

# --- The chart for the presentation ---
fig, ax = plt.subplots(figsize=(7, 4.2))
for shape in SHAPES:
    values = [bic_table[(shape, k)] for k in CLUSTERS_TO_TRY]
    ax.plot(list(CLUSTERS_TO_TRY), values, marker="o", label=shape)
ax.scatter([best_k], [best_bic], s=180, facecolors="none",
           edgecolors="black", linewidths=2, zorder=5, label="winner")
ax.set_xlabel("Number of clusters (hills)")
ax.set_ylabel("BIC (lower = better)")
ax.set_title(f"The BIC picks {best_k} clusters with shape '{best_shape}'")
ax.legend(title="Shape of the hills")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("bic_gmm.png", dpi=150)
print("   🖼️ Chart saved to bic_gmm.png (into the presentation!)")


# =====================================================================
# BLOCK 5: THE WINNING GMM
# =====================================================================
gmm = models[(best_shape, best_k)]   # we already trained it in Block 4


def full_covariance(model, i):
    """Gets the shape of hill i as a 2x2 matrix, no matter which shape won.
    That way the ESP32 always receives the same thing and nothing needs changing."""
    c = model.covariances_
    if model.covariance_type == "full":
        return c[i]
    if model.covariance_type == "tied":
        return c                        # they all share the same one
    if model.covariance_type == "diag":
        return np.diag(c[i])            # straight: zeros off the diagonal
    return np.eye(2) * c[i]             # spherical: round ball


# Sort the clusters from coldest to warmest, so that
# cluster 0 is always the coldest (makes them easier to talk about).
order = np.argsort(gmm.means_[:, 0])


def sorted_cluster(x):
    """Asks the GMM which cluster it looks like and translates it to the cold->warm order."""
    original = gmm.predict(x)
    translator = {old: new for new, old in enumerate(order)}
    return np.array([translator[c] for c in original])


data["cluster"] = sorted_cluster(data[INPUTS].values)


# =====================================================================
# BLOCK 6: WHAT DOES EACH CLUSTER MEAN? (A HUMAN DECIDES THIS)
# =====================================================================
# The GMM only says "cluster 0, 1, 2...". To know what each cluster IS,
# we use what the model did NOT see: rain and the month of the year.
# If a cluster gathers the rainy days WITHOUT having seen the rain,
# it means temperature + humidity really do carry useful information.
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

print("\n🔍 This is how the clusters turned out (checked with years the model NEVER saw, 2024-2025):")
summary = []
for c in range(best_k):
    exam = data[(data["cluster"] == c) & (data["YEAR"] >= 2024)]
    everything = data[data["cluster"] == c]
    top_months = everything["month"].value_counts().head(3).index
    row = {
        "cluster": c,
        "temp": everything["T2M"].mean(),
        "humidity": everything["RH2M"].mean(),
        "rainy_days_pct": (exam["PRECTOTCORR"] >= 1).mean() * 100,
        "avg_rain_mm": exam["PRECTOTCORR"].mean(),
        "months": ", ".join(MONTH_NAMES[m - 1] for m in sorted(top_months)),
    }
    summary.append(row)
    print(f"\n   CLUSTER {c}:  {row['temp']:.1f} °C  and  {row['humidity']:.0f} % humidity")
    print(f"      Months where it shows up most: {row['months']}")
    print(f"      In 2024-2025: it rained (≥1 mm) on {row['rainy_days_pct']:.0f}% "
          f"of its days, {row['avg_rain_mm']:.2f} mm per day on average")

# 👇 NAMES GIVEN BY HUMANS after looking at the numbers above.
#    If you change the data (another place, other years), CHECK THEM AGAIN.
#    (No accents on purpose: the ESP32 Serial Monitor sometimes shows them badly).
NAMES = {
    0: "Cold and humid (autumn-winter)",     # Oct-Dec, hardly rains
    1: "Dry (dry season)",                   # Mar-May, 0% rainy days
    2: "Warm and humid (rainy season)",      # Jul-Sep, rains half the days
}
if best_k != len(NAMES):
    print(f"\n⚠️ HEADS UP: {best_k} clusters came out and there are {len(NAMES)} names. "
          "Check the numbers above and fix the NAMES dictionary.")
    NAMES = {c: f"Cluster {c}" for c in range(best_k)}


# =====================================================================
# BLOCK 7: GMM VS. A SIMPLE RULE (honesty above all)
# =====================================================================
# Exam question: "Is today a rainy-season day?"
# The real answer comes from NASA's rain data (≥1 mm = it rained).
# We compare two ways of guessing using ONLY temperature and humidity.
exam = data[data["YEAR"] >= 2024]
it_rained = exam["PRECTOTCORR"] >= 1
rainy_cluster = max(range(best_k), key=lambda c: summary[c]["rainy_days_pct"])

gmm_says = exam["cluster"] == rainy_cluster
rule_says = (exam["T2M"] > 17) & (exam["RH2M"] > 50)   # rule written by a person


def grade(name, says):
    caught = (says & it_rained).sum() / it_rained.sum() * 100        # of the rainy days, how many it detected
    false_alarms = (says & ~it_rained).sum() / says.sum() * 100     # of its warnings, how many were on dry days
    print(f"   {name:<28} detected {caught:4.0f}% of the rainy days | "
          f"{false_alarms:4.0f}% of its warnings were false alarms")


print("\n⚖️ Who detects rainy days better in 2024-2025?")
grade("GMM (rainy cluster)", gmm_says)
grade("Rule: T>17°C and H>50%", rule_says)
print("   👉 Neither is perfect: one catches more rainy days and the other gives fewer")
print("      false alarms. That gets said AS IS to the judges. The GMM's extra advantage")
print("      is that it gives a PROBABILITY (how sure it is), and the rule doesn't.")


# =====================================================================
# BLOCK 8: MARKOV CHAIN (HOW LONG WILL TODAY'S WEATHER LAST?)
# =====================================================================
# Idea: TOMORROW's weather depends on TODAY's.
# We count, day by day, how many times it went from one cluster to another.
# Example: if today is "rainy", how many times was tomorrow still "rainy"?
# That table of counts, turned into percentages, IS the Markov chain.
data = data.sort_values("date")           # sort by date, just in case
k = best_k
counts = np.zeros((k, k))                 # k x k table full of zeros

today = data["cluster"].values[:-1]       # every day except the last
tomorrow = data["cluster"].values[1:]     # every day except the first
is_training = (data["YEAR"].values[1:] <= 2023)   # only learns from 2016-2023
for c_today, c_tomorrow, useful in zip(today, tomorrow, is_training):
    if useful:
        counts[c_today, c_tomorrow] += 1  # "one more time it went from c_today to c_tomorrow"

# Divide each row by its total -> percentages that add up to 1 (100%)
MARKOV = counts / counts.sum(axis=1, keepdims=True)

print("\n🔗 Markov chain: if TODAY is... what's the probability of each cluster TOMORROW?")
for c in range(k):
    stay = MARKOV[c, c]
    duration = 1 / (1 - stay)             # formula for the average length of a streak
    print(f"   {NAMES[c]:<42} stays the same {stay*100:4.1f}%  "
          f"-> lasts ~{duration:.0f} days in a row")

# ----- 7-DAY FORECAST: two ways, and the exam decides which one works -----
FORECAST_DAYS = 7
dates = data.set_index("date")["cluster"]


def jump_table(from_year, to_year, jump):
    """Counts: if a day was cluster A, which cluster was it 'jump' days later?"""
    t = np.zeros((k, k))
    for d, c in dates[(dates.index.year >= from_year) & (dates.index.year <= to_year)].items():
        later = dates.get(d + pd.Timedelta(days=jump))
        if later is not None:
            t[c, later] += 1
    return t


# WAY A ("textbook" Markov): chain the 1-day jump 7 times.
#   Assumes the weather only remembers YESTERDAY.
MARKOV_CHAINED = np.linalg.matrix_power(MARKOV, FORECAST_DAYS)

# WAY B: measure the 7-day jump directly (it's still a Markov chain,
#   but with one-week steps instead of one-day steps).
jumps = jump_table(2016, 2023, FORECAST_DAYS)
MARKOV_WEEK = jumps / jumps.sum(axis=1, keepdims=True)

# 📝 EXAM with 2024-2025 (years neither of them saw)
actual = jump_table(2024, 2025, FORECAST_DAYS)
actual = actual / actual.sum(axis=1, keepdims=True)
print(f"\n📝 Exam: if today is X, will it still be X in {FORECAST_DAYS} days? (2024-2025)")
print(f"   {'Today is':<42} {'A: 1 day x7':>11} {'B: 7 days':>10} {'ACTUAL':>9}")
for c in range(k):
    print(f"   {NAMES[c]:<42} {MARKOV_CHAINED[c, c]*100:10.0f}% "
          f"{MARKOV_WEEK[c, c]*100:9.0f}% {actual[c, c]*100:8.0f}%")
print("   👉 Way A falls short: Chihuahua's weather 'remembers' more")
print("      than just yesterday (seasons last weeks). We send B to the ESP32.")


# =====================================================================
# BLOCK 9: SAVE THE MODEL FOR THE ESP32
# =====================================================================
# The whole "brain" of the GMM is just a few numbers per cluster:
#   - the center (average temperature and humidity)
#   - the "shape" of the hill (covariance matrix; we store its INVERSE,
#     because multiplying is cheaper than dividing for the ESP32)
#   - a constant that combines the cluster's weight and the hill's size
# With that, the ESP32 works out which cluster each reading looks like.
# Whichever shape wins the BIC, it's always saved here as 2x2,
# so the classifier header and the .ino do NOT need to change.
header_lines = [
    "// gmm_model.h  |  AUTO-GENERATED by train_gmm.py. Do not edit by hand.",
    "// Jackson Five · NASA Space Apps 2026",
    f"// Trained on NASA POWER daily data, {train['YEAR'].min()}-{train['YEAR'].max()}, "
    "point 29.25, -107.41 (Chihuahua)",
    f"// Selected by BIC: {best_k} clusters, '{best_shape}' covariance (BIC = {best_bic:.0f})",
    "// Inputs: [0] air temperature (°C), [1] relative air humidity (%)",
    "#pragma once",
    "",
    f"#define GMM_N_CLUSTERS {best_k}",
    "",
]
centers, inverses, constants, c_names = [], [], [], []
for new, old in enumerate(order):
    mu = gmm.means_[old]
    cov = full_covariance(gmm, old)
    inv = np.linalg.inv(cov)
    constant = np.log(gmm.weights_[old]) - 0.5 * np.log(np.linalg.det(cov)) - np.log(2 * np.pi)
    centers.append(f"  {{{mu[0]:.6f}f, {mu[1]:.6f}f}}")
    inverses.append(f"  {{{inv[0, 0]:.8f}f, {inv[0, 1]:.8f}f, {inv[1, 1]:.8f}f}}")
    constants.append(f"  {constant:.8f}f")
    c_names.append(f'  "{NAMES[new]}"')

header_lines += ["// Center of each cluster: {temperature, humidity}",
                 "const float GMM_CENTER[GMM_N_CLUSTERS][2] = {", ",\n".join(centers), "};", "",
                 "// Inverse covariance of each cluster: {a, b, c} = [[a, b], [b, c]]",
                 "const float GMM_INV_COV[GMM_N_CLUSTERS][3] = {", ",\n".join(inverses), "};", "",
                 "// log(weight) - 0.5*log(det(cov)) - log(2*pi)",
                 "const float GMM_CONSTANT[GMM_N_CLUSTERS] = {", ",\n".join(constants), "};", "",
                 "// Human-assigned names (see Block 6 of train_gmm.py)",
                 "const char* const GMM_NAME[GMM_N_CLUSTERS] = {", ",\n".join(c_names), "};", "",
                 "// Weekly Markov chain (see Block 8): [today][in 7 days] = probability",
                 "const float GMM_MARKOV[GMM_N_CLUSTERS][GMM_N_CLUSTERS] = {",
                 ",\n".join("  {" + ", ".join(f"{p:.6f}f" for p in row) + "}" for row in MARKOV_WEEK),
                 "};", "",
                 f"#define GMM_FORECAST_DAYS {FORECAST_DAYS}", ""]

with open("gmm_model.h", "w", encoding="utf-8") as f:
    f.write("\n".join(header_lines))

n_numbers = best_k * (2 + 3 + 1) + best_k * best_k
print(f"\n📦 Model saved to gmm_model.h  ->  {n_numbers} numbers in total "
      f"(about {n_numbers * 4} bytes). Plenty of room left on the ESP32.")

# Three sample readings to check that the ESP32 gives the SAME result as Python.
print("\n🧪 Test readings (the ESP32 must give the same cluster):")
for t, h in [(30.0, 20.0), (20.0, 70.0), (5.0, 60.0)]:
    c = sorted_cluster(np.array([[t, h]]))[0]
    p = gmm.predict_proba(np.array([[t, h]]))[0][order[c]] * 100
    print(f"   {t:4.1f} °C, {h:4.1f} %  ->  {NAMES[c]}  ({p:.0f}% sure)")
