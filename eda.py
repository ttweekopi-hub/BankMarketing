from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency, mannwhitneyu


st.set_page_config(
    page_title="Bank Marketing EDA",
    page_icon="",
    layout="wide",
)

PAGES = [
    "Overview",
    "Dataset Information",
    "Data Quality and Target Distribution",
    "Numerical Feature Analysis and Target Comparisons",
    "Categorical Feature Analysis and Target Comparisons",
    "Statistical Associations and Correlation Review",
    "Leakage and Modelling Risks",
    "Conclusions and Modelling Strategy",
]

DATA_DICTIONARY = pd.DataFrame(
    [
        {"column": "age", "group": "Client profile", "description": "Client age.", "type": "numeric"},
        {"column": "job", "group": "Client profile", "description": "Type of job.", "type": "categorical"},
        {"column": "marital", "group": "Client profile", "description": "Marital status.", "type": "categorical"},
        {"column": "education", "group": "Client profile", "description": "Education level.", "type": "categorical"},
        {"column": "default", "group": "Client profile", "description": "Whether the client has credit in default.", "type": "categorical"},
        {"column": "housing", "group": "Client profile", "description": "Whether the client has a housing loan.", "type": "categorical"},
        {"column": "loan", "group": "Client profile", "description": "Whether the client has a personal loan.", "type": "categorical"},
        {"column": "contact", "group": "Current campaign", "description": "Contact communication type.", "type": "categorical"},
        {"column": "month", "group": "Current campaign", "description": "Last contact month of year.", "type": "categorical"},
        {"column": "day_of_week", "group": "Current campaign", "description": "Last contact day of week.", "type": "categorical"},
        {"column": "duration", "group": "Current campaign", "description": "Last contact duration in seconds.", "type": "numeric"},
        {"column": "campaign", "group": "Current campaign", "description": "Number of contacts during this campaign.", "type": "numeric"},
        {"column": "pdays", "group": "Previous campaign", "description": "Days since previous contact; 999 means not previously contacted.", "type": "numeric"},
        {"column": "previous", "group": "Previous campaign", "description": "Number of contacts before this campaign.", "type": "numeric"},
        {"column": "poutcome", "group": "Previous campaign", "description": "Outcome of the previous marketing campaign.", "type": "categorical"},
        {"column": "emp.var.rate", "group": "Economic context", "description": "Employment variation rate, quarterly indicator.", "type": "numeric"},
        {"column": "cons.price.idx", "group": "Economic context", "description": "Consumer price index, monthly indicator.", "type": "numeric"},
        {"column": "cons.conf.idx", "group": "Economic context", "description": "Consumer confidence index, monthly indicator.", "type": "numeric"},
        {"column": "euribor3m", "group": "Economic context", "description": "Euribor 3 month rate, daily indicator.", "type": "numeric"},
        {"column": "nr.employed", "group": "Economic context", "description": "Number of employees, quarterly indicator.", "type": "numeric"},
        {"column": "y", "group": "Target", "description": "Whether the client subscribed to a term deposit.", "type": "binary"},
    ]
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = Path("data/bank-additional-full.csv")
    return pd.read_csv(data_path)


@st.cache_data
def add_target_flag(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(target_yes=data["y"].map({"no": 0, "yes": 1}))


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help=help_text)


def section_note(title: str, body: str) -> None:
    st.markdown(f"#### {title}")
    st.write(body)


def subscription_rate_table(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    summary = (
        data.groupby(feature)
        .agg(count=("target_yes", "size"), subscription_rate_percent=("target_yes", "mean"))
        .reset_index()
    )
    summary["subscription_rate_percent"] = (summary["subscription_rate_percent"] * 100).round(2)
    return summary.sort_values("subscription_rate_percent", ascending=False)


def bar_subscription_rate(data: pd.DataFrame, feature: str, title: str):
    rate = subscription_rate_table(data, feature)
    fig = px.bar(
        rate,
        x="subscription_rate_percent",
        y=feature,
        orientation="h",
        text="subscription_rate_percent",
        color="subscription_rate_percent",
        color_continuous_scale="Teal",
        title=title,
        hover_data={"count": True, "subscription_rate_percent": ":.2f"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Subscription rate (%)",
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        height=max(360, 28 * len(rate) + 160),
        margin=dict(l=10, r=30, t=70, b=35),
    )
    return fig


def unknown_summary(data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = data.select_dtypes(include="object").columns
    counts = data[categorical_columns].eq("unknown").sum().sort_values(ascending=False)
    summary = pd.DataFrame(
        {
            "column": counts.index,
            "unknown_count": counts.values,
            "unknown_percent": (counts.values / len(data) * 100).round(2),
        }
    )
    return summary[summary["unknown_count"] > 0]


def outlier_summary(data: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in numeric_columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        rows.append(
            {
                "column": column,
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outlier_count": int(outliers.sum()),
                "outlier_percent": round(outliers.mean() * 100, 2),
                "minimum": data[column].min(),
                "maximum": data[column].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_percent", ascending=False)


def chi_square_summary(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        contingency_table = pd.crosstab(data[feature], data["y"])
        chi2_statistic, p_value, degrees_of_freedom, _ = chi2_contingency(contingency_table)
        n = contingency_table.to_numpy().sum()
        min_dimension = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2_statistic / (n * min_dimension))
        rows.append(
            {
                "feature": feature,
                "chi2_statistic": round(chi2_statistic, 2),
                "p_value": p_value,
                "degrees_of_freedom": degrees_of_freedom,
                "cramers_v": round(cramers_v, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)


def mann_whitney_summary(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        no_values = data.loc[data["y"] == "no", feature]
        yes_values = data.loc[data["y"] == "yes", feature]
        statistic, p_value = mannwhitneyu(no_values, yes_values, alternative="two-sided")
        rows.append(
            {
                "feature": feature,
                "median_no": no_values.median(),
                "median_yes": yes_values.median(),
                "mann_whitney_u": round(statistic, 2),
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


df = load_data()
eda_df = add_target_flag(df)
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = df.select_dtypes(include="object").columns.tolist()
target_counts = df["y"].value_counts()
target_percent = df["y"].value_counts(normalize=True) * 100
yes_rate = target_percent.get("yes", 0)

st.sidebar.title("Bank Marketing EDA")
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.caption("Storytelling view based on the EDA notebook findings.")

st.title("Bank Marketing Term Deposit EDA")

if page == "Overview":
    st.subheader("The campaign has a small subscriber group and several strong modelling signals.")
    st.write(
        """
        This application turns the EDA notebook into a guided story for the Portuguese bank marketing dataset.
        The goal is to understand who subscribed to a term deposit, what patterns separate subscribers from
        non-subscribers, and what risks must be handled before modelling.
        """
    )

    metric_row(
        [
            ("Records", f"{len(df):,}", None),
            ("Columns", f"{df.shape[1]:,}", None),
            ("Subscription rate", f"{yes_rate:.2f}%", "Share of records where y = yes."),
            ("Duplicate rows", f"{df.duplicated().sum():,}", None),
        ]
    )

    target_chart = px.pie(
        values=target_counts.reindex(["no", "yes"]).values,
        names=["No subscription", "Subscribed"],
        hole=0.45,
        color=["No subscription", "Subscribed"],
        color_discrete_map={"No subscription": "#216205", "Subscribed": "#08011c"},
        title="Target distribution",
    )
    target_chart.update_traces(textinfo="label+percent", sort=False)
    st.plotly_chart(target_chart, width="stretch")

    section_note(
        "How to read the story",
        """
        The analysis starts with data quality and target imbalance, then moves into numerical and categorical
        patterns, statistical association checks, leakage risks, and finally a modelling strategy. The main
        modelling message is that accuracy alone is not enough because only 11.27% of clients subscribed.
        """,
    )

    st.dataframe(DATA_DICTIONARY, hide_index=True, width="stretch")

elif page == "Dataset Information":
    st.subheader("The dataset mixes customer profile, campaign activity, previous campaign history, and economic context.")
    metric_row(
        [
            ("Rows", f"{df.shape[0]:,}", None),
            ("Columns", f"{df.shape[1]:,}", None),
            ("Numerical columns", f"{len(numeric_columns):,}", None),
            ("Categorical columns", f"{len(categorical_columns):,}", None),
        ]
    )

    st.markdown("#### First rows")
    st.dataframe(df.head(10), width="stretch")

    st.markdown("#### Data types and missing values")
    column_info = pd.DataFrame(
        {
            "column": df.columns,
            "non_null_count": df.notna().sum().values,
            "missing_count": df.isna().sum().values,
            "dtype": df.dtypes.astype(str).values,
        }
    )
    st.dataframe(column_info, hide_index=True, width="stretch")

    st.markdown("#### Feature groups")
    st.write(
        """
        - **Client profile:** `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
        - **Current campaign:** `contact`, `month`, `day_of_week`, `duration`, `campaign`
        - **Previous campaign:** `pdays`, `previous`, `poutcome`
        - **Economic context:** `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
        - **Target:** `y`
        """
    )

    section_note(
        "Dataset takeaway",
        """
        The target column is binary, so the future modelling task is a classification problem. The mixture
        of categorical and numerical inputs means preprocessing will need both category encoding and numeric
        handling.
        """,
    )

elif page == "Data Quality and Target Distribution":
    st.subheader("There are no standard nulls, but hidden missingness and imbalance matter.")
    missing_summary = df.isna().sum()
    quality_summary = pd.DataFrame(
        {
            "check": ["Columns with standard missing values", "Duplicate rows", "Columns with unknown category"],
            "result": [
                int((missing_summary > 0).sum()),
                int(df.duplicated().sum()),
                int(len(unknown_summary(df))),
            ],
        }
    )
    st.dataframe(quality_summary, hide_index=True, width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Hidden missingness")
        st.dataframe(unknown_summary(df), hide_index=True, width="stretch")
        st.write(
            """
            The largest hidden missingness issue is `default`, followed by `education`, `housing`, `loan`,
            `job`, and `marital`. These values are text labels, so normal null checks do not catch them.
            """
        )

    with right:
        st.markdown("#### Target distribution")
        target_summary = pd.DataFrame(
            {
                "target": target_counts.index,
                "count": target_counts.values,
                "percent": target_percent.round(2).values,
            }
        )
        st.dataframe(target_summary, hide_index=True, width="stretch")
        target_bar = px.bar(
            target_summary,
            x="target",
            y="count",
            text="percent",
            color="target",
            color_discrete_map={"no": "#6b7280", "yes": "#0f766e"},
            title="Chart 1: Target Distribution",
        )
        target_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        target_bar.update_layout(showlegend=False, xaxis_title="Subscribed", yaxis_title="Records")
        st.plotly_chart(target_bar, width="stretch")

    duplicate_records = df[df.duplicated(keep=False)]
    with st.expander("View duplicate records"):
        st.write(f"{len(duplicate_records):,} rows are involved in {df.duplicated().sum():,} duplicate records.")
        st.dataframe(duplicate_records, width="stretch")

    section_note(
        "Why this matters",
        """
        A model that predicts every client as `no` would be about 88.73% accurate but would miss all subscribers.
        Future evaluation should therefore focus on recall, precision, F1-score, ROC-AUC, PR-AUC, and the
        confusion matrix instead of relying on accuracy alone.
        """,
    )

elif page == "Numerical Feature Analysis and Target Comparisons":
    st.subheader("Numerical features show skew, outliers, and clear target-group differences.")
    st.dataframe(df[numeric_columns].describe().T.round(2), width="stretch")

    selected_numeric = st.selectbox(
        "Select a numerical feature",
        ["age", "duration", "campaign", "pdays", "previous", "euribor3m", "nr.employed"],
    )

    left, right = st.columns([1, 1])
    with left:
        fig = px.histogram(
            df,
            x=selected_numeric,
            nbins=40,
            marginal="box",
            color="y",
            color_discrete_map={"no": "#6b7280", "yes": "#0f766e"},
            title=f"Distribution of {selected_numeric}",
        )
        fig.update_layout(yaxis_title="Records")
        st.plotly_chart(fig, width="stretch")
    with right:
        box = px.box(
            df,
            x="y",
            y=selected_numeric,
            color="y",
            category_orders={"y": ["no", "yes"]},
            color_discrete_map={"no": "#6b7280", "yes": "#0f766e"},
            title=f"{selected_numeric} compared with target",
        )
        box.update_layout(showlegend=False, xaxis_title="Subscribed")
        st.plotly_chart(box, width="stretch")

    numeric_findings = {
        "age": "Most clients are in their 30s and 40s. Subscribers are slightly more spread across older age groups.",
        "duration": "Subscribers generally had longer calls, but this is a major leakage risk if prediction happens before the call ends.",
        "campaign": "Most clients were contacted only a few times. Very high contact counts are uncommon and mostly appear among non-subscribers.",
        "pdays": "Most records have `pdays = 999`, which means not previously contacted rather than 999 actual days.",
        "previous": "Most clients had no previous campaign contact. Previous contact history becomes more useful when read with `poutcome`.",
        "euribor3m": "Lower Euribor values appear more common among subscribers, suggesting macroeconomic context is related to outcomes.",
        "nr.employed": "Lower employment indicator values appear more common among subscribers and overlap with other economic indicators.",
    }
    st.info(numeric_findings[selected_numeric])

    st.markdown("#### Special value and outlier review")
    pdays_999_percent = df["pdays"].eq(999).mean() * 100
    metric_row(
        [
            ("Rows with pdays = 999", f"{pdays_999_percent:.2f}%", "999 means not previously contacted."),
            ("Mean duration, no", f"{df.loc[df['y'] == 'no', 'duration'].mean():.2f}s", None),
            ("Mean duration, yes", f"{df.loc[df['y'] == 'yes', 'duration'].mean():.2f}s", None),
        ]
    )
    st.dataframe(outlier_summary(df, numeric_columns), hide_index=True, width="stretch")

    st.markdown("#### Grouped means by target")
    st.dataframe(df.groupby("y")[["duration", "campaign", "previous", "euribor3m"]].mean().round(2), width="stretch")

elif page == "Categorical Feature Analysis and Target Comparisons":
    st.subheader("Campaign timing, channel, previous outcome, and customer profile all change subscription rates.")

    primary_feature = st.selectbox(
        "Select a categorical feature",
        ["job", "education", "contact", "month", "poutcome", "marital", "default", "housing", "loan", "day_of_week"],
    )
    st.plotly_chart(
        bar_subscription_rate(eda_df, primary_feature, f"Subscription rate by {primary_feature}"),
        width="stretch",
    )
    st.dataframe(subscription_rate_table(eda_df, primary_feature), hide_index=True, width="stretch")

    section_note(
        "Key categorical findings",
        """
        Students and retired clients have higher subscription rates among job groups. Cellular contact performs
        better than telephone contact. March, December, September, and October stand out among campaign months.
        Previous campaign success is one of the clearest positive signals.
        """,
    )

    st.markdown("#### Campaign interaction heatmaps")
    interaction_choice = st.radio(
        "Interaction view",
        ["Previous outcome x contact type", "Previous outcome x contact month"],
        horizontal=True,
    )
    if interaction_choice == "Previous outcome x contact type":
        pivot = eda_df.pivot_table(index="poutcome", columns="contact", values="target_yes", aggfunc="mean") * 100
    else:
        pivot = eda_df.pivot_table(index="poutcome", columns="month", values="target_yes", aggfunc="mean") * 100

    heatmap = px.imshow(
        pivot.round(1),
        text_auto=".1f",
        color_continuous_scale="Teal",
        aspect="auto",
        title=f"Subscription rate (%): {interaction_choice}",
    )
    heatmap.update_layout(xaxis_title="", yaxis_title="Previous outcome")
    st.plotly_chart(heatmap, width="stretch")

    st.write(
        """
        The interaction view shows that previous campaign outcome remains important, but the strength of that
        signal changes by contact type and month. These patterns are useful candidates for modelling, but should
        be validated rather than treated as causal explanations.
        """
    )

elif page == "Statistical Associations and Correlation Review":
    st.subheader("Statistical checks support the visual patterns, while correlations warn about overlapping economic indicators.")
    categorical_test_features = [
        "job",
        "education",
        "contact",
        "month",
        "poutcome",
        "marital",
        "default",
        "housing",
        "loan",
        "day_of_week",
    ]
    numerical_test_features = ["age", "duration", "campaign", "pdays", "previous", "euribor3m"]
    economic_columns = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Categorical association tests")
        chi_summary = chi_square_summary(df, categorical_test_features)
        st.dataframe(chi_summary, hide_index=True, width="stretch")
        st.write(
            """
            Cramer's V is more useful than p-value alone here because the dataset is large. `poutcome`,
            `contact`, and `month` show stronger categorical association with the target.
            """
        )
    with right:
        st.markdown("#### Numerical target-group tests")
        mann_summary = mann_whitney_summary(df, numerical_test_features)
        st.dataframe(mann_summary, hide_index=True, width="stretch")
        st.write(
            """
            Mann-Whitney U tests are used because several numerical variables are skewed. The test supports
            the notebook finding that numerical distributions differ between subscribers and non-subscribers.
            """
        )

    st.markdown("#### Economic indicators by target")
    st.dataframe(df.groupby("y")[economic_columns].mean().round(2), width="stretch")
    economic_feature = st.selectbox("Select an economic indicator", economic_columns)
    econ_box = px.box(
        df,
        x="y",
        y=economic_feature,
        color="y",
        category_orders={"y": ["no", "yes"]},
        color_discrete_map={"no": "#6b7280", "yes": "#0f766e"},
        title=f"{economic_feature} by target",
    )
    econ_box.update_layout(showlegend=False, xaxis_title="Subscribed")
    st.plotly_chart(econ_box, width="stretch")

    st.markdown("#### Correlation heatmap")
    corr = df[numeric_columns].corr().round(2)
    corr_fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Heatmap of Numerical Features",
    )
    st.plotly_chart(corr_fig, width="stretch")
    st.write(
        """
        `emp.var.rate`, `euribor3m`, and `nr.employed` move closely together. This is not necessarily a problem
        for tree-based prediction, but it does mean feature-importance explanations should avoid overstating one
        economic variable when it overlaps strongly with others.
        """
    )

elif page == "Leakage and Modelling Risks":
    st.subheader("The strongest-looking features are not automatically safe modelling features.")

    risk_table = pd.DataFrame(
        [
            {
                "risk": "Call duration leakage",
                "feature": "duration",
                "why_it_matters": "Duration is known only after the call ends, so it may not be available before a call is made.",
                "modelling_action": "Train and report models both with and without duration, then align the chosen model to the real prediction timing.",
            },
            {
                "risk": "Special value treated as a real number",
                "feature": "pdays",
                "why_it_matters": "999 means the client was not previously contacted, not that 999 days passed.",
                "modelling_action": "Create a `was_previously_contacted` flag and handle actual day values separately.",
            },
            {
                "risk": "Hidden missingness",
                "feature": "unknown categories",
                "why_it_matters": "`unknown` may encode unavailable information and is not captured by null checks.",
                "modelling_action": "Keep `unknown` explicitly or compare against an imputation strategy during preprocessing.",
            },
            {
                "risk": "Imbalanced target",
                "feature": "y",
                "why_it_matters": "Only 11.27% of records are subscribers, so accuracy can look good while recall is poor.",
                "modelling_action": "Use recall, precision, F1-score, ROC-AUC, PR-AUC, and confusion matrices.",
            },
            {
                "risk": "Correlated economic indicators",
                "feature": "emp.var.rate, euribor3m, nr.employed",
                "why_it_matters": "Overlapping macroeconomic variables can make interpretation unstable.",
                "modelling_action": "Monitor correlation and avoid causal or over-specific importance claims.",
            },
        ]
    )
    st.dataframe(risk_table, hide_index=True, width="stretch")

    st.markdown("#### Duration effect versus leakage concern")
    duration_summary = df.groupby("y")["duration"].agg(["count", "median", "mean", "max"]).round(2)
    st.dataframe(duration_summary, width="stretch")
    st.write(
        """
        `duration` has a strong relationship with subscription because successful calls tend to last longer.
        The same strength is exactly why it must be treated carefully: it can make a model look excellent while
        answering a prediction question that the business cannot actually use before the call is completed.
        """
    )

    st.markdown("#### `pdays` special handling")
    pdays_summary = pd.DataFrame(
        [
            {"group": "Not previously contacted (`pdays = 999`)", "rows": int(df["pdays"].eq(999).sum()), "percent": round(df["pdays"].eq(999).mean() * 100, 2)},
            {"group": "Previously contacted (`pdays != 999`)", "rows": int(df["pdays"].ne(999).sum()), "percent": round(df["pdays"].ne(999).mean() * 100, 2)},
        ]
    )
    st.dataframe(pdays_summary, hide_index=True, width="stretch")

elif page == "Conclusions and Modelling Strategy":
    st.subheader("The EDA supports a leakage-aware, imbalance-aware classification strategy.")

    st.markdown(
        """
        #### Key findings
        1. The target is imbalanced: only **11.27%** of clients subscribed.
        2. The dataset has no standard nulls, but several categorical columns contain hidden `unknown` values.
        3. Numerical features such as `duration`, `campaign`, and `previous` are skewed and contain high-end outliers.
        4. `pdays = 999` needs special handling because it means the client was not previously contacted.
        5. Categorical features and interactions show useful differences in subscription rates, especially `poutcome`, `contact`, and `month`.
        6. Statistical checks support the visual findings, but large-sample p-values should be interpreted with effect sizes.
        7. Economic variables differ by target group and are also correlated with each other.
        """
    )

    strategy = pd.DataFrame(
        [
            {
                "step": "Baseline",
                "recommendation": "Start with Logistic Regression.",
                "reason": "It is fast, interpretable, and confirms whether preprocessing works.",
            },
            {
                "step": "Non-linear model",
                "recommendation": "Compare against Random Forest.",
                "reason": "It can capture interactions between customer, campaign, and economic features.",
            },
            {
                "step": "High-performance candidate",
                "recommendation": "Evaluate Gradient Boosted Trees.",
                "reason": "Boosted trees usually perform well on structured tabular classification tasks.",
            },
            {
                "step": "Leakage check",
                "recommendation": "Run models with and without `duration`.",
                "reason": "The right version depends on whether prediction happens before or after the call.",
            },
            {
                "step": "Feature engineering",
                "recommendation": "Test `was_previously_contacted`, age bins, debt indicator, and month encodings.",
                "reason": "These ideas come from EDA but should be validated through cross-validation.",
            },
            {
                "step": "Evaluation",
                "recommendation": "Prioritise recall, precision, F1-score, PR-AUC, ROC-AUC, and confusion matrix.",
                "reason": "Accuracy is misleading when the positive class is small.",
            },
        ]
    )
    st.dataframe(strategy, hide_index=True, width="stretch")

    st.success(
        """
        Recommended modelling path: build a simple, reproducible preprocessing pipeline first; benchmark Logistic
        Regression; compare tree-based models; report performance with and without `duration`; and explain results
        using metrics that reflect the imbalanced subscriber class.
        """
    )
