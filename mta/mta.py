import pandas as pd
from itertools import chain, tee, combinations
from functools import reduce, wraps
from operator import mul
from collections import defaultdict, Counter
import random
import time
import numpy as np
import copy
import os
import sys
import math
from collections.abc import Callable
from typing import List, Any, Dict, Tuple, DefaultDict, Optional, Union
from dataclasses import dataclass
import arrow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class MTAConfig:
    """Configuration for MTA model"""

    allow_loops: bool = False
    add_timepoints: bool = True
    sep: str = " > "
    normalize_by_default: bool = True


def show_time(func: Callable[..., Any]):
    """Timer decorator"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        print(f"running {func.__name__}.. ", end="")
        sys.stdout.flush()

        v = func(*args, **kwargs)

        minutes, seconds = divmod(time.time() - t0, 60)
        st = "elapsed time:"
        if minutes:
            st += f" {minutes:.0f} min"
        if seconds:
            st += f" {seconds:.3f} sec"
        print(st)

        return v

    return wrapper


class MTA:
    """Multi-Touch Attribution model implementation"""

    def __init__(
        self,
        data: Union[str, pd.DataFrame] = "data.csv.gz",
        config: Optional[MTAConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize MTA model

        Args:
            data: Path to CSV file or DataFrame
            config: MTAConfig object or use kwargs for individual settings
        """
        # Setup configuration
        if config is None:
            config = MTAConfig(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in MTAConfig.__dataclass_fields__
                }
            )
        self.config = config

        # Constants
        self.NULL = "(null)"
        self.START = "(start)"
        self.CONV = "(conversion)"

        # Load and validate data
        self._load_data(data)
        self._validate_data()

        # Process data
        if config.add_timepoints:
            self.add_exposure_times()
        if not config.allow_loops:
            self.remove_loops()

        self._prepare_data()
        self._setup_channels()

        # Initialize results storage
        self.attribution: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def _load_data(self, data: Union[str, pd.DataFrame]) -> None:
        """Load data from file or DataFrame"""
        if isinstance(data, str):
            data_path = os.path.join(os.path.dirname(__file__), "data", data)
            self.data = pd.read_csv(data_path)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise TypeError("data must be a file path or pandas DataFrame")

    def _validate_data(self) -> None:
        """Validate required columns exist"""
        required_cols = {"path", "total_conversions", "total_null"}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(
                f"Data must contain columns: {required_cols}. "
                f"Found: {set(self.data.columns)}"
            )

    def _prepare_data(self) -> None:
        """Convert path and exposure_times to lists"""
        self.data[["path", "exposure_times"]] = self.data[
            ["path", "exposure_times"]
        ].map(lambda x: [ch.strip() for ch in str(x).split(self.config.sep.strip())])

    def _setup_channels(self) -> None:
        """Setup channel mappings and indices"""
        self.channels = sorted({ch for ch in chain.from_iterable(self.data["path"])})
        self.channels_ext = [self.START] + self.channels + [self.CONV, self.NULL]
        self.channel_name_to_index = {c: i for i, c in enumerate(self.channels_ext)}
        self.index_to_channel_name = {
            i: c for c, i in self.channel_name_to_index.items()
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(channels={len(self.channels)}, "
            f"journeys={len(self.data)})"
        )

    def add_exposure_times(self, exposure_every_second: bool = True) -> "MTA":
        """Generate synthetic exposure times"""
        if "exposure_times" in self.data.columns:
            return self

        ts = []
        _t0 = arrow.utcnow()

        if exposure_every_second:
            for path_str in self.data["path"]:
                path_list = [ch.strip() for ch in path_str.split(">")]
                time_range = arrow.Arrow.range(
                    "second", _t0, _t0.shift(seconds=len(path_list) - 1)
                )
                ts.append(
                    self.config.sep.join(
                        [t.format("YYYY-MM-DD HH:mm:ss") for t in time_range]
                    )
                )

        self.data["exposure_times"] = ts
        return self

    @show_time
    def remove_loops(self) -> "MTA":
        """Remove consecutive duplicate channels"""
        cleaned_data = []

        for _, row in self.data.iterrows():
            path = [ch.strip() for ch in str(row["path"]).split(">")]
            exposure = [ch.strip() for ch in str(row["exposure_times"]).split(">")]

            clean_path, clean_exposure = [], []
            prev_channel = None

            for ch, exp in zip(path, exposure):
                if ch != prev_channel:
                    clean_path.append(ch)
                    clean_exposure.append(exp)
                    prev_channel = ch

            cleaned_data.append(
                {
                    "path": self.config.sep.join(clean_path),
                    "exposure_times": self.config.sep.join(clean_exposure),
                    **{
                        col: row[col]
                        for col in self.data.columns
                        if col not in ["path", "exposure_times"]
                    },
                }
            )

        self.data = pd.DataFrame(cleaned_data)

        # Aggregate duplicate paths
        agg_dict = {
            col: "sum"
            for col in self.data.columns
            if col not in ["path", "exposure_times"]
        }
        agg_dict["exposure_times"] = "first"

        self.data = self.data.groupby("path", as_index=False).agg(agg_dict)
        return self

    @staticmethod
    def normalize_dict(
        dict_: Dict[Any, float], decimal_digits: int = 6
    ) -> Dict[Any, float]:
        """Normalize dictionary values to sum to 1"""
        total = sum(dict_.values())
        return (
            {k: round(v / total, decimal_digits) for k, v in dict_.items()}
            if total
            else dict_
        )

    def _apply_normalization(
        self, result: Dict[str, float], normalize: bool
    ) -> Dict[str, float]:
        """Apply normalization if requested"""
        return self.normalize_dict(result) if normalize else result

    @show_time
    def linear(self, share: str = "same", normalize: bool = True) -> "MTA":
        """
        Linear attribution model

        Args:
            share: 'same' for equal distribution or 'proportional' for weighted
            normalize: Whether to normalize results
        """
        if share not in ["same", "proportional"]:
            raise ValueError("share must be 'same' or 'proportional'")

        linear = defaultdict(float)

        for row in self.data.itertuples():
            if not row.total_conversions:
                continue

            if share == "same":
                unique_channels = set(row.path)
                credit_per_channel = row.total_conversions / len(unique_channels)
                for c in unique_channels:
                    linear[c] += credit_per_channel
            else:  # proportional
                channel_counts = Counter(row.path)
                total_touches = sum(channel_counts.values())
                for c, count in channel_counts.items():
                    linear[c] += row.total_conversions * (count / total_touches)

        linear = self._apply_normalization(dict(linear), normalize)
        self.attribution["linear"] = linear
        return self

    @show_time
    def position_based(
        self, first_weight: float = 40, last_weight: float = 40, normalize: bool = True
    ) -> "MTA":
        """
        Position-based attribution

        Args:
            first_weight: Percentage credit for first touch (0-100)
            last_weight: Percentage credit for last touch (0-100)
            normalize: Whether to normalize results
        """
        if first_weight + last_weight > 100:
            raise ValueError("Sum of first and last weights cannot exceed 100")

        position_based = defaultdict(float)

        for row in self.data.itertuples():
            if not row.total_conversions:
                continue

            path_len = len(row.path)

            if path_len == 1:
                position_based[row.path[0]] += row.total_conversions
            elif path_len == 2:
                credit = row.total_conversions / 2
                position_based[row.path[0]] += credit
                position_based[row.path[-1]] += credit
            else:
                position_based[row.path[0]] += (
                    first_weight * row.total_conversions / 100
                )
                position_based[row.path[-1]] += (
                    last_weight * row.total_conversions / 100
                )

                middle_credit = (
                    (100 - first_weight - last_weight) * row.total_conversions / 100
                )
                middle_channels = row.path[1:-1]
                credit_per_middle = middle_credit / len(middle_channels)

                for c in middle_channels:
                    position_based[c] += credit_per_middle

        position_based = self._apply_normalization(dict(position_based), normalize)
        self.attribution["pos_based"] = position_based
        return self

    @show_time
    def time_decay(
        self, count_direction: str = "left", normalize: bool = True
    ) -> "MTA":
        """
        Time decay attribution - channels closer to conversion get more credit

        Args:
            count_direction: 'left' (oldest first) or 'right' (newest first)
            normalize: Whether to normalize results
        """
        if count_direction not in ["left", "right"]:
            raise ValueError("count_direction must be 'left' or 'right'")

        time_decay = defaultdict(float)

        for row in self.data.itertuples():
            if not row.total_conversions:
                continue

            # Get unique channels in order of appearance
            seen = []
            path_to_iterate = (
                row.path if count_direction == "left" else reversed(row.path)
            )

            for c in path_to_iterate:
                if c not in seen:
                    seen.append(c)

            if count_direction == "right":
                seen.reverse()

            # Assign weights: 1, 2, 3, ... (linear growth)
            total_weight = sum(range(1, len(seen) + 1))

            for i, c in enumerate(seen, 1):
                time_decay[c] += (i / total_weight) * row.total_conversions

        time_decay = self._apply_normalization(dict(time_decay), normalize)
        self.attribution["time_decay"] = time_decay
        return self

    @show_time
    def first_touch(self, normalize: bool = True) -> "MTA":
        """First-touch attribution model"""
        first_touch = (
            self.data[self.data["total_conversions"] > 0]
            .groupby(self.data["path"].apply(lambda x: x[0]))["total_conversions"]
            .sum()
            .to_dict()
        )

        first_touch = self._apply_normalization(first_touch, normalize)
        self.attribution["first_touch"] = first_touch
        return self

    @show_time
    def last_touch(self, normalize: bool = True) -> "MTA":
        """Last-touch attribution model"""
        last_touch = (
            self.data[self.data["total_conversions"] > 0]
            .groupby(self.data["path"].apply(lambda x: x[-1]))["total_conversions"]
            .sum()
            .to_dict()
        )

        last_touch = self._apply_normalization(last_touch, normalize)
        self.attribution["last_touch"] = last_touch
        return self

    @staticmethod
    def pairs(lst: List[Any]) -> zip:
        """Generate consecutive pairs from list"""
        it1, it2 = tee(lst)
        next(it2, None)
        return zip(it1, it2)

    def count_pairs(self) -> DefaultDict[Tuple[str, str], int]:
        """Count channel pair transitions"""
        pair_counts = defaultdict(int)

        for row in self.data.itertuples():
            # Count transitions along the path
            for pair in self.pairs([self.START] + row.path):
                pair_counts[pair] += row.total_conversions + row.total_null

            # Add terminal transitions
            pair_counts[(row.path[-1], self.NULL)] += row.total_null
            pair_counts[(row.path[-1], self.CONV)] += row.total_conversions

        return pair_counts

    def ordered_tuple(self, t: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """
        Return tuple t ordered
        Special case: if tuple starts with START, keep it first and sort the rest
        """
        if len(t) > 1 and t[0] == self.START:
            return (t[0],) + tuple(sorted(t[1:]))
        return tuple(sorted(t))

    def transition_matrix(self) -> Dict[Tuple[str, str], float]:
        """Calculate Markov transition probabilities"""
        pair_counts = self.count_pairs()

        # Calculate outgoing transition totals
        outgoing = defaultdict(int)
        for (from_ch, _), count in pair_counts.items():
            outgoing[from_ch] += count

        # Calculate probabilities
        return {pair: count / outgoing[pair[0]] for pair, count in pair_counts.items()}

    @show_time
    def simulate_path(
        self,
        trans_mat: Dict[Tuple[str, str], float],
        drop_channel: Optional[str] = None,
        n: int = int(1e6),
    ) -> Dict[str, int]:
        """Simulate random user journeys using Markov chain"""
        outcome_counts = defaultdict(int)

        idx_start = self.channel_name_to_index[self.START]
        idx_null = self.channel_name_to_index[self.NULL]
        idx_conv = self.channel_name_to_index[self.CONV]
        idx_drop = self.channel_name_to_index.get(drop_channel, idx_null)

        for _ in range(n):
            current_idx = idx_start

            while True:
                # Get transition probabilities from current state
                probs = [
                    trans_mat.get(
                        (
                            self.index_to_channel_name[current_idx],
                            self.index_to_channel_name[i],
                        ),
                        0,
                    )
                    for i in range(len(self.channels_ext))
                ]

                # Choose next state
                next_idx = np.random.choice(len(self.channels_ext), p=probs)

                if next_idx == idx_conv:
                    outcome_counts[self.CONV] += 1
                    break
                elif next_idx in {idx_null, idx_drop}:
                    outcome_counts[self.NULL] += 1
                    break
                else:
                    current_idx = next_idx

        return dict(outcome_counts)

    def _calculate_path_probability(
        self, path: List[str], trans_mat: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate probability of a specific path"""
        full_path = [self.START] + path + [self.CONV]
        probs = [trans_mat.get(pair, 0) for pair in self.pairs(full_path)]
        return reduce(mul, probs) if probs else 0

    def prob_convert(
        self, trans_mat: Dict[Tuple[str, str], float], drop: Optional[str] = None
    ) -> float:
        """Calculate total conversion probability"""
        # Filter data
        if drop:
            mask = ~self.data["path"].apply(lambda x: drop in x) & (
                self.data["total_conversions"] > 0
            )
        else:
            mask = self.data["total_conversions"] > 0

        filtered_data = self.data[mask]

        # Sum probabilities across all converting paths
        total_prob = sum(
            self._calculate_path_probability(row.path, trans_mat)
            for row in filtered_data.itertuples()
        )

        return total_prob

    @show_time
    def markov(self, sim: bool = False, normalize: bool = True) -> "MTA":
        """
        Markov chain attribution model

        Args:
            sim: Use simulation (True) or analytical calculation (False)
            normalize: Whether to normalize results
        """
        markov = defaultdict(float)
        trans_mat = self.transition_matrix()

        if sim:
            # Simulation-based approach
            outcomes_full = self.simulate_path(trans_mat, drop_channel=None)

            for channel in self.channels:
                outcomes_drop = self.simulate_path(trans_mat, drop_channel=channel)
                markov[channel] = (
                    outcomes_full[self.CONV] - outcomes_drop[self.CONV]
                ) / outcomes_full[self.CONV]
        else:
            # Analytical approach
            p_base = self.prob_convert(trans_mat)

            for channel in self.channels:
                p_without = self.prob_convert(trans_mat, drop=channel)
                markov[channel] = (p_base - p_without) / p_base if p_base else 0

        markov = self._apply_normalization(dict(markov), normalize)
        self.attribution["markov"] = markov
        return self

    @show_time
    def shao(self, normalize: bool = True) -> "MTA":
        """
        probabilistic model by Shao and Li (supposed to be equivalent to Shapley); explanation in the original paper may seem rather unclear but
        this https://stats.stackexchange.com/questions/255312/multi-channel-attribution-modelling-using-a-simple-probabilistic-model
        is definitely helpful
        """

        r = defaultdict(lambda: defaultdict(float))

        # count user conversions and nulls for each visited channel and channel pair

        for row in self.data.itertuples():

            for n in range(1, 3):

                # # combinations('ABCD', 2) --> AB AC AD BC BD CD

                for ch in combinations(set(row.path), n):

                    t = self.ordered_tuple(ch)

                    r[t][self.CONV] += row.total_conversions
                    r[t][self.NULL] += row.total_null

        for _ in r:
            r[_]["conv_prob"] = r[_][self.CONV] / (r[_][self.CONV] + r[_][self.NULL])

        # calculate channel contributions

        self.C = defaultdict(float)

        for row in self.data.itertuples():

            for ch_i in set(row.path):

                if row.total_conversions:

                    pc = 0  # contribution for current path

                    other_channels = set(row.path) - {ch_i}

                    k = 2 * len(other_channels) if other_channels else 1

                    for ch_j in other_channels:

                        pc += (
                            r[self.ordered_tuple((ch_i, ch_j))]["conv_prob"]
                            - r[(ch_i,)]["conv_prob"]
                            - r[(ch_j,)]["conv_prob"]
                        )

                    pc = r[(ch_i,)]["conv_prob"] + pc / k

                    self.C[ch_i] += row.total_conversions * pc

        if normalize:
            self.C = self.normalize_dict(self.C)

        self.attribution["shao"] = self.C

        return self

    def get_generated_conversions(self, max_subset_size: int = 3) -> "MTA":  # FIXED

        self.cc = defaultdict(lambda: defaultdict(float))

        for ch_list, convs, nulls in zip(
            self.data["path"], self.data["total_conversions"], self.data["total_null"]
        ):

            # only look at journeys with conversions
            for n in range(1, max_subset_size + 1):

                for tup in combinations(set(ch_list), n):

                    tup_ = self.ordered_tuple(tup)

                    self.cc[tup_][self.CONV] += convs
                    self.cc[tup_][self.NULL] += nulls

        return self

    def v(self, coalition: Tuple[Any, Any]) -> float:
        """
        total number of conversions generated by all subsets of the coalition;
        coalition is a tuple of channels
        """

        s = len(coalition)

        total_convs = 0

        for n in range(1, s + 1):
            for tup in combinations(coalition, n):
                tup_ = self.ordered_tuple(tup)
                total_convs += self.cc[tup_][self.CONV]

        return total_convs

    def w(self, s, n):

        # FIXED: Handle edge cases properly
        # Formula: s! * (n - s - 1)! / n!
        if s >= n or s < 0:
            return 0
        return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)

    @show_time
    def shapley(self, max_coalition_size: int = 2, normalize: bool = True) -> "MTA":
        """
        Shapley value attribution

        Args:
            max_coalition_size: Maximum size of coalitions to consider
            normalize: Whether to normalize results
        """
        self.get_generated_conversions(max_subset_size=3)

        shapley = defaultdict(float)
        n_channels = len(self.channels)

        for channel in self.channels:
            other_channels = set(self.channels) - {channel}

            # Consider all subset sizes up to max_coalition_size
            for subset_size in range(
                1, min(max_coalition_size + 1, len(other_channels) + 1)
            ):
                for coalition in combinations(other_channels, subset_size):
                    marginal_contribution = self.v(coalition + (channel,)) - self.v(
                        coalition
                    )
                    weight = self.w(len(coalition), n_channels)
                    shapley[channel] += marginal_contribution * weight

        shapley = self._apply_normalization(dict(shapley), normalize)
        self.attribution["shapley"] = shapley
        return self

    @show_time
    def logistic_regression(
        self,
        test_size: float = 0.25,
        sample_rows: float = 0.5,
        sample_features: float = 0.5,
        normalize: bool = True,
        n_iterations: int = 2000,
    ) -> "MTA":
        """
        Logistic regression attribution using ensemble approach

        Args:
            test_size: Proportion for test set
            sample_rows: Proportion of rows to sample each iteration
            sample_features: Proportion of features to sample each iteration
            normalize: Whether to normalize results
            n_iterations: Number of bootstrap iterations
        """
        # Build feature matrix
        records = []
        for row in self.data.itertuples():
            channel_set = {c: 1 for c in row.path}

            for _ in range(row.total_conversions):
                records.append({**channel_set, "conv": 1})

            for _ in range(row.total_null):
                records.append({**channel_set, "conv": 0})

        df = pd.DataFrame(records).fillna(0).sample(frac=1.0, random_state=42)

        # Ensemble learning
        coef_sum = defaultdict(float)

        for i in range(n_iterations):
            # Balanced sampling
            df_conv = df[df["conv"] == 1].sample(frac=0.5, random_state=i)
            df_null = df[df["conv"] == 0].sample(frac=0.5, random_state=i)
            df_sample = pd.concat([df_conv, df_null]).sample(
                frac=sample_rows, random_state=i
            )

            # Feature sampling
            feature_cols = (
                df_sample.drop("conv", axis=1)
                .sample(frac=sample_features, axis=1, random_state=i)
                .columns
            )

            X = df_sample[feature_cols]
            y = df_sample["conv"]

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=i, stratify=y
            )

            clf = LogisticRegression(
                random_state=i, solver="lbfgs", fit_intercept=False, max_iter=1000
            ).fit(X_train, y_train)

            # Accumulate coefficients
            for channel, coef in zip(X_train.columns, clf.coef_[0]):
                coef_sum[channel] += abs(coef)

        # Average coefficients
        lr_attribution = {ch: coef / n_iterations for ch, coef in coef_sum.items()}

        lr_attribution = self._apply_normalization(lr_attribution, normalize)
        self.attribution["linreg"] = lr_attribution
        return self

    def show(self, channels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Display attribution results

        Args:
            channels: Specific channels to show (None for all)

        Returns:
            DataFrame with attribution results
        """
        df = pd.DataFrame.from_dict(self.attribution)

        if channels:
            df = df.loc[df.index.isin(channels)]

        # Sort by index (channel name)
        df = df.sort_index()

        print("\nAttribution Results:")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80)

        return df

    def compare_models(self) -> pd.DataFrame:
        """Compare all attribution models side by side"""
        df = self.show()

        # Add summary statistics
        print("\nModel Statistics:")
        print(df.describe())

        return df

    def export_results(self, filepath: str, format: str = "csv") -> None:
        """
        Export attribution results

        Args:
            filepath: Output file path
            format: 'csv', 'json', or 'excel'
        """
        df = pd.DataFrame.from_dict(self.attribution)

        if format == "csv":
            df.to_csv(filepath)
        elif format == "json":
            df.to_json(filepath, orient="index", indent=2)
        elif format == "excel":
            df.to_excel(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Results exported to {filepath}")

    def pi(
        self, path, exposure_times, conv_flag, beta_by_channel, omega_by_channel
    ) -> Dict[str, float]:
        """

        calculate contribution of channel i to conversion of journey (user) u - (p_i^u) in the paper

         - path is a list of states that includes (start) but EXCLUDES (null) or (conversion)
         - exposure_times is list of exposure times

        """

        p = {c: 0 for c in path}  # contributions by channel

        # all contributions are zero if no conversion
        if not conv_flag:
            return p

        dts = [
            (arrow.get(exposure_times[-1]) - arrow.get(t)).seconds
            for t in exposure_times
        ]

        _ = defaultdict(float)

        for c, dt in zip(path, dts):
            _[c] += (
                beta_by_channel[c]
                * omega_by_channel[c]
                * np.exp(-omega_by_channel[c] * dt)
            )

        for c in _:
            p[c] = _[c] / sum(_.values())

        return p

    def update_coefs(self, beta: float, omega: float) -> Tuple[float, float, float]:
        """
        return updated beta and omega
        """

        delta = 1e-3

        beta_num = defaultdict(float)
        beta_den = defaultdict(float)
        omega_den = defaultdict(float)

        for u, row in enumerate(self.data.itertuples()):

            p = self.pi(
                row.path, row.exposure_times, row.total_conversions, beta, omega
            )

            r = copy.deepcopy(row.path)

            dts = [
                (arrow.get(row.exposure_times[-1]) - arrow.get(t)).seconds
                for t in row.exposure_times
            ]

            while r:

                # pick channels starting from the last one
                c = r.pop()
                dt = dts.pop()

                beta_den[c] += 1.0 - np.exp(-omega[c] * dt)
                omega_den[c] += p[c] * dt + beta[c] * dt * np.exp(-omega[c] * dt)

                beta_num[c] += p[c]

        # now that we gone through every user, update coefficients for every channel

        beta0 = copy.deepcopy(beta)
        omega0 = copy.deepcopy(omega)

        df = []

        for c in self.channels:

            beta_num[c] = (beta_num[c] > 1e-6) * beta_num[c]
            beta_den[c] = (beta_den[c] > 1e-6) * beta_den[c]
            omega_den[c] = max(omega_den[c], 1e-6)

            if beta_den[c]:
                beta[c] = beta_num[c] / beta_den[c]

            omega[c] = beta_num[c] / omega_den[c]

            df.append(abs(beta[c] - beta0[c]) < delta)
            df.append(abs(omega[c] - omega0[c]) < delta)

        return (beta, omega, sum(df))

    @show_time
    def additive_hazard(
        self, epochs: int = 20, normalize: bool = True
    ) -> "MTA":  # FIXED
        """
        additive hazard model as in Multi-Touch Attribution in On-line Advertising with Survival Theory
        """

        beta = {c: random.uniform(0.001, 1) for c in self.channels}
        omega = {c: random.uniform(0.001, 1) for c in self.channels}

        for _ in range(epochs):

            beta, omega, h = self.update_coefs(beta, omega)

            if h == 2 * len(self.channels):
                print(f"converged after {_ + 1} iterations")
                break

        # time window: take the max time instant across all journeys that converged

        additive_hazard = defaultdict(float)

        for u, row in enumerate(self.data.itertuples()):

            p = self.pi(
                row.path, row.exposure_times, row.total_conversions, beta, omega
            )

            for c in p:
                # FIXED: Weight by actual conversions
                additive_hazard[c] += p[c] * row.total_conversions

        if normalize:
            additive_hazard = self.normalize_dict(additive_hazard)

        self.attribution["add_haz"] = additive_hazard

        return self


if __name__ == "__main__":

    mta = MTA(data="data.csv.gz", allow_loops=False)

    (
        mta.linear(share="proportional")
        .time_decay(count_direction="right")
        .shapley()
        .shao()
        .first_touch()
        .position_based()
        .last_touch()
        .markov(sim=False)
        .logistic_regression()
        .additive_hazard()
        .show()
    )
