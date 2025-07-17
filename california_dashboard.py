
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="California Housing Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("california_house_price.csv")
    # Feature engineering (same as notebook)
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    df['lat_bin'] = pd.cut(df['latitude'], bins=5)
    df['age_bin'] = pd.cut(df['housing_median_age'], bins=range(0, 55, 5))
    df['age_category'] = df['housing_median_age'].apply(lambda x: 'New (<20)' if x < 20 else 'Old (>=20)')
    return df

housing = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Age slider
age_min, age_max = int(housing['housing_median_age'].min()), int(housing['housing_median_age'].max())
age_range = st.sidebar.slider("Housing Age (years)", age_min, age_max, (age_min, age_max))

# Median Income slider
inc_min, inc_max = float(housing['median_income'].min()), float(housing['median_income'].max())
income_range = st.sidebar.slider("Median Income (10k$)", inc_min, inc_max, (inc_min, inc_max))

# Rooms per household slider
rph_min, rph_max = float(housing['rooms_per_household'].min()), float(min(10, housing['rooms_per_household'].max()))
rph_range = st.sidebar.slider("Rooms per Household", rph_min, rph_max, (rph_min, rph_max))

# Ocean proximity multiselect
ocean_options = housing['ocean_proximity'].unique().tolist()
selected_ocean = st.sidebar.multiselect("Ocean Proximity", options=ocean_options, default=ocean_options)

# Apply filters
filtered = housing[
    (housing['housing_median_age'].between(*age_range)) &
    (housing['median_income'].between(*income_range)) &
    (housing['rooms_per_household'].between(*rph_range)) &
    (housing['ocean_proximity'].isin(selected_ocean))
]

st.title("California Housing Interactive Dashboard")

# KPI cards
k1, k2, k3 = st.columns(3)
k1.metric("Avg House Value", f"${filtered['median_house_value'].mean():,.0f}")
k2.metric("Avg Income (10k$)", f"{filtered['median_income'].mean():.2f}")
k3.metric("Filtered Rows", f"{len(filtered):,}")

# --- Tabs for plots ---
tab_names = [
    "House Value Distribution", "Income Distribution", "Income vs Value",
    "Value by Ocean Proximity", "Latitude Bin vs Value", "Rooms/HH vs Value",
    "Bedrooms/Room vs Value", "Population vs Value", "Correlation Heatmap",
    "Geographical Map", "Value by Age Group"
]
tabs = st.tabs(tab_names)

# 1. House value distribution
with tabs[0]:
    st.subheader("Distribution of Median House Value")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(filtered['median_house_value'], bins=40, color='skyblue')
    ax.set_xlabel("Median House Value ($)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# 2. Income distribution
with tabs[1]:
    st.subheader("Distribution of Median Income")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(filtered['median_income'], bins=40, color='salmon')
    ax.set_xlabel("Median Income (10k$)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# 3. Income vs Value scatter
with tabs[2]:
    st.subheader("Median Income vs Median House Value")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(filtered['median_income'], filtered['median_house_value'], alpha=0.4)
    ax.set_xlabel("Median Income (10k$)")
    ax.set_ylabel("Median House Value ($)")
    st.pyplot(fig)

# 4. Box plot by ocean proximity
with tabs[3]:
    st.subheader("House Value by Ocean Proximity")
    fig, ax = plt.subplots(figsize=(10,6))
    cats = filtered['ocean_proximity'].unique()
    data = [filtered[filtered['ocean_proximity']==c]['median_house_value'] for c in cats]
    ax.boxplot(data, tick_labels=cats)
    ax.set_xlabel("Ocean Proximity")
    ax.set_ylabel("Median House Value ($)")
    st.pyplot(fig)

# 5. Latitude bin bar  (Interval → str 로 변환해 그리기)
with tabs[4]:
    st.subheader("Average House Value by Latitude Bin")

    lat_value = (
        filtered
        .groupby('lat_bin', observed=False)['median_house_value']
        .mean()
        .reset_index()                       # DataFrame 으로 변환
    )
    lat_value['lat_bin'] = lat_value['lat_bin'].astype(str)   # ← 문자열 변환

    # Streamlit bar_chart는 인덱스를 x축으로 쓰므로 set_index 후 호출
    st.bar_chart(
        lat_value.set_index('lat_bin')
    )


# 6. Rooms per household vs value
with tabs[5]:
    st.subheader("Rooms per Household vs Median House Value")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(filtered['rooms_per_household'], filtered['median_house_value'], alpha=0.4)
    ax.set_xlabel("Rooms per Household")
    ax.set_ylabel("Median House Value ($)")
    st.pyplot(fig)

# 7. Bedrooms per room vs value
with tabs[6]:
    st.subheader("Bedrooms per Room vs Median House Value")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(filtered['bedrooms_per_room'], filtered['median_house_value'], alpha=0.4)
    ax.set_xlabel("Bedrooms per Room")
    ax.set_ylabel("Median House Value ($)")
    st.pyplot(fig)

# 8. Population vs value
with tabs[7]:
    st.subheader("Population vs Median House Value")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(filtered['population'], filtered['median_house_value'], alpha=0.3)
    ax.set_xlabel("Population")
    ax.set_ylabel("Median House Value ($)")
    st.pyplot(fig)

# 9. Correlation heatmap
with tabs[8]:
    st.subheader("Correlation Heatmap (numeric variables)")
    num_cols = ['median_house_value', 'median_income', 'total_rooms', 'total_bedrooms', 'population',
                'households', 'housing_median_age', 'rooms_per_household', 'bedrooms_per_room',
                'population_per_household']
    corr = filtered[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='right')
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

# 10. Geo scatter
with tabs[9]:
    st.subheader("Geographical Distribution of House Prices")
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(filtered['longitude'], filtered['latitude'],
                    c=filtered['median_house_value'], cmap='viridis', alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Median House Value ($)')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

# 11. Age category boxplot
with tabs[10]:
    st.subheader("House Prices by Age Category")
    fig, ax = plt.subplots(figsize=(8,5))
    cats = ['New (<20)', 'Old (>=20)']
    data = [filtered[filtered['age_category']==c]['median_house_value'] for c in cats]
    ax.boxplot(data, tick_labels=cats)
    ax.set_xlabel("Age Category")
    ax.set_ylabel("Median House Value ($)")
    st.pyplot(fig)
