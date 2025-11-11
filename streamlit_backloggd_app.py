# streamlit_backloggd_app.py
# Streamlit 앱 — backloggd_games.csv를 읽고 인터랙티브한 EDA 및 시각화, 필터, 다운로드 기능을 제공합니다.
# 사용법:
# 1) 필요한 패키지 설치: pip install streamlit pandas numpy plotly matplotlib altair
# 2) 실행: streamlit run streamlit_backloggd_app.py

import streamlit as st
import pandas as pd
import numpy as np
import ast
import io
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# 유틸리티 함수들
# -------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data
def parse_numeric_shorthand(x: Any) -> float | None:
    # '21K' -> 21000, '3.2K' -> 3200, '1M' -> 1_000_000, '—' or NaN -> None
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == '' or s == '—' or s.lower() == 'nan':
        return None
    s = s.replace(',', '')
    try:
        if s[-1].upper() == 'K':
            return float(s[:-1]) * 1_000
        if s[-1].upper() == 'M':
            return float(s[:-1]) * 1_000_000
        return float(s)
    except Exception:
        return None

@st.cache_data
def safe_eval_list(x: Any) -> List[str]:
    # 문자열로 되어 있는 리스트 표현을 안전하게 파싱한다. 실패하면 빈 리스트 반환
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(i) for i in parsed]
        # 만약 단일 문자열이 콤마로 구분되어 있으면 분해
        if ',' in s:
            return [p.strip() for p in s.split(',') if p.strip()]
        return [s]
    except Exception:
        # fallback: comma split
        if ',' in s:
            return [p.strip() for p in s.split(',') if p.strip()]
        return [s]

@st.cache_data
def extract_year(release_date: Any) -> int | None:
    if pd.isna(release_date):
        return None
    s = str(release_date)
    # 일반적인 형태: 'Feb 25, 2022' 또는 '2022-02-25' 등
    for fmt in ('%b %d, %Y', '%B %d, %Y', '%Y-%m-%d', '%Y'):
        try:
            return datetime.strptime(s, fmt).year
        except Exception:
            pass
    # 숫자 4자리 검색
    import re
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return int(m.group(0))
    return None

# -------------------------
# 데이터 전처리 함수
# -------------------------
@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 표준 컬럼 이름 만들기 (있다면)
    colmap = {c: c.strip() for c in df.columns}
    df.columns = [colmap[c] for c in df.columns]

    # 기본 칼럼이 없을 수 있으므로 확인
    expected = ['Title', 'Release_Date', 'Developers', 'Summary', 'Platforms', 'Genres', 'Rating',
                'Plays', 'Playing', 'Backlogs', 'Wishlist', 'Lists', 'Reviews']

    # numeric shorthand 변환
    for col in ['Plays', 'Playing', 'Backlogs', 'Wishlist', 'Lists', 'Reviews']:
        if col in df.columns:
            df[col + '_num'] = df[col].apply(parse_numeric_shorthand)
        else:
            df[col + '_num'] = None

    # Rating 결측은 그대로 둠
    if 'Rating' in df.columns:
        # ensure float
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    else:
        df['Rating'] = np.nan

    # Genres, Platforms, Developers 파싱
    for col in ['Genres', 'Platforms', 'Developers']:
        if col in df.columns:
            df[col + '_list'] = df[col].apply(safe_eval_list)
        else:
            df[col + '_list'] = [[] for _ in range(len(df))]

    # Release year
    if 'Release_Date' in df.columns:
        df['Release_Year'] = df['Release_Date'].apply(extract_year)
    else:
        df['Release_Year'] = None

    return df

# -------------------------
# 분석/시각화 helper
# -------------------------
@st.cache_data
def top_n_by(col: str, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df[['Title', col]].dropna().sort_values(by=col, ascending=False).head(n)

@st.cache_data
def genre_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    # 장르별 평균 평점 및 카운트
    genre_map: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        genres = row.get('Genres_list', [])
        rating = row.get('Rating', np.nan)
        for g in genres:
            if g not in genre_map:
                genre_map[g] = {'ratings': [], 'count': 0}
            if not np.isnan(rating):
                genre_map[g]['ratings'].append(rating)
            genre_map[g]['count'] += 1
    out = []
    for g, v in genre_map.items():
        avg = np.mean(v['ratings']) if v['ratings'] else np.nan
        out.append({'Genre': g, 'Average_Rating': avg, 'Count': v['count']})
    return pd.DataFrame(out).sort_values(by='Count', ascending=False)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title='Backloggd Games EDA', layout='wide', initial_sidebar_state='expanded')

st.title('🎮 Backloggd Games — 인터랙티브 데이터 분석 대시보드')
st.markdown(
    "업로드된 CSV 파일을 분석하고 시각화합니다. 기본적으로 `/mnt/data/backloggd_games.csv` (서버 경로)가 있으면 불러옵니다."
)

# 사이드바: 파일 업로드 또는 기본 파일 사용
st.sidebar.header('데이터 입력')
uploaded_file = st.sidebar.file_uploader('CSV 파일 업로드 (backloggd_games.csv 권장)', type=['csv'])
use_default = False
if uploaded_file is None:
    # 기본 경로를 먼저 시도
    default_path = '/mnt/data/backloggd_games.csv'
    try:
        df_raw = load_csv(default_path)
        use_default = True
        st.sidebar.write(f'기본 파일을 사용합니다: `{default_path}`')
    except Exception:
        st.sidebar.info('기본 파일을 찾을 수 없습니다. 파일을 업로드해주세요.')
        df_raw = None
else:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.success('파일 업로드 완료')

if df_raw is None:
    st.stop()

# 전처리
with st.spinner('데이터 전처리 중...'):
    df = preprocess(df_raw)

# 상단 KPI
st.header('요약 지표')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('총 게임 수', f"{len(df):,}")
with col2:
    avg_rating = df['Rating'].mean()
    st.metric('평균 평점', f"{avg_rating:.2f}" if not np.isnan(avg_rating) else 'N/A')
with col3:
    total_plays = df['Plays_num'].dropna().sum()
    st.metric('전체 플레이 합계', f"{int(total_plays):,}" if not np.isnan(total_plays) else 'N/A')
with col4:
    earliest = df['Release_Year'].dropna().min()
    latest = df['Release_Year'].dropna().max()
    st.metric('출시 연도 범위', f"{int(earliest)} — {int(latest)}" if not np.isnan(earliest) and not np.isnan(latest) else 'N/A')

# 필터 패널
st.sidebar.header('필터')
min_rating = st.sidebar.slider('최소 평점', 0.0, 5.0, 3.5, 0.1)
year_range = st.sidebar.slider('출시 연도 범위', int(df['Release_Year'].dropna().min()) if df['Release_Year'].dropna().any() else 2000,
                               int(df['Release_Year'].dropna().max()) if df['Release_Year'].dropna().any() else 2025,
                               (int(df['Release_Year'].dropna().min()) if df['Release_Year'].dropna().any() else 2000,
                                int(df['Release_Year'].dropna().max()) if df['Release_Year'].dropna().any() else 2025))

# 장르 멀티선택
all_genres = sorted({g for L in df['Genres_list'] for g in L if g})
selected_genres = st.sidebar.multiselect('장르 선택 (빈칸이면 전체)', all_genres, default=None)

# 플랫폼 다중 선택
all_platforms = sorted({p for L in df['Platforms_list'] for p in L if p})
selected_platforms = st.sidebar.multiselect('플랫폼 선택 (빈칸이면 전체)', all_platforms, default=None)

# 적용 필터
filtered = df.copy()
filtered = filtered[filtered['Rating'].fillna(0) >= min_rating]
filtered = filtered[filtered['Release_Year'].apply(lambda y: y is not None and year_range[0] <= y <= year_range[1])]
if selected_genres:
    filtered = filtered[filtered['Genres_list'].apply(lambda lst: any(g in lst for g in selected_genres))]
if selected_platforms:
    filtered = filtered[filtered['Platforms_list'].apply(lambda lst: any(p in lst for p in selected_platforms))]

st.sidebar.write(f'필터 적용 후 게임 수: {len(filtered):,}')

# 탭: 개요, 장르, 개발사, 상호작용 차트, 데이터
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Overview', 'Genres', 'Developers', 'Interactive Charts', 'Data'])

with tab1:
    st.subheader('평점 분포')
    fig = px.histogram(filtered, x='Rating', nbins=30, title='평점 분포 (필터 적용)')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('출시 연도별 게임 수')
    year_counts = filtered['Release_Year'].dropna().astype(int).value_counts().sort_index()
    if not year_counts.empty:
        fig2 = px.bar(x=year_counts.index, y=year_counts.values, labels={'x':'Year','y':'Count'}, title='출시 연도별 게임 수')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info('출시 연도 데이터가 부족합니다.')

    st.subheader('평점 상위 게임들')
    st.dataframe(filtered[['Title', 'Release_Date', 'Rating', 'Plays_num', 'Backlogs_num']].sort_values(by='Rating', ascending=False).head(20))

with tab2:
    st.subheader('장르별 통계')
    ga = genre_aggregations(df)
    st.dataframe(ga.head(50))

    st.markdown('---')
    st.subheader('선택 장르별 평점 비교')
    if selected_genres:
        comp = ga[ga['Genre'].isin(selected_genres)]
        figg = px.bar(comp, x='Genre', y='Average_Rating', title='선택 장르별 평균 평점')
        st.plotly_chart(figg, use_container_width=True)
    else:
        st.info('사이드바에서 장르를 선택하면 해당 장르들의 비교를 보여줍니다.')

with tab3:
    st.subheader('개발사별 상위 (평균 평점 기준)')
    # developers_list는 리스트들 중 첫 개발사에 집중하거나 모든 개발사를 풀어냄
    dev_map = {}
    for _, row in df.iterrows():
        devs = row['Developers_list']
        rating = row['Rating']
        title = row['Title']
        for d in devs:
            if d not in dev_map:
                dev_map[d] = {'ratings': [], 'titles': []}
            if not np.isnan(rating):
                dev_map[d]['ratings'].append(rating)
            dev_map[d]['titles'].append(title)
    dev_df = pd.DataFrame([
        {'Developer': d, 'Average_Rating': np.mean(v['ratings']) if v['ratings'] else np.nan, 'Game_Count': len(v['titles'])}
        for d, v in dev_map.items()
    ])
    dev_df = dev_df.sort_values(by='Game_Count', ascending=False)
    st.dataframe(dev_df.head(50))

with tab4:
    st.subheader('상호작용형 스캐터 — 플레이 수 vs 평점')
    scatter_df = filtered.dropna(subset=['Plays_num', 'Rating'])
    if not scatter_df.empty:
        fig = px.scatter(scatter_df, x='Plays_num', y='Rating', hover_data=['Title', 'Release_Year'],
                         title='Plays vs Rating (로그 스케일 선택 가능)')
        log_x = st.checkbox('X 축 로그 스케일 적용', value=True)
        if log_x:
            fig.update_layout(xaxis_type='log')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Plays 또는 Rating 데이터가 부족합니다.')

    st.markdown('---')
    st.subheader('상관행렬 (숫자형 변수)')
    num_cols = [c for c in df.columns if c.endswith('_num')] + ['Rating']
    corr = df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, title='숫자형 변수 상관행렬')
    st.plotly_chart(fig_corr, use_container_width=True)

with tab5:
    st.subheader('원본 데이터 미리보기')
    st.dataframe(df.head(200))

    st.markdown('---')
    st.subheader('필터 적용 결과 다운로드')
    to_download = filtered.copy()
    # 필요한 컬럼만 정리
    csv_buf = to_download.to_csv(index=False).encode('utf-8')
    st.download_button('필터된 CSV 다운로드', data=csv_buf, file_name='backloggd_filtered.csv', mime='text/csv')

# 추가: 사용자 정의 쿼리 영역
st.sidebar.header('빠른 검색')
search_title = st.sidebar.text_input('게임 제목 검색 (부분 일치)')
if search_title:
    res = df[df['Title'].str.contains(search_title, case=False, na=False)]
    st.sidebar.write(f'검색 결과: {len(res)}개')
    if st.sidebar.checkbox('검색 결과 보기'):
        st.write(res[['Title', 'Release_Date', 'Rating', 'Plays_num']].head(50))

# 하단: 앱 정보
with st.expander('앱 정보 / 요구사항'):
    st.markdown(
        """
        **필요 패키지**:
        - streamlit
        - pandas
        - numpy
        - plotly
        - matplotlib
        
        설치 예시:
        ```bash
        pip install streamlit pandas numpy plotly matplotlib
        streamlit run streamlit_backloggd_app.py
        ```
        
        **설명**:
        - 사이드바에서 필터(평점, 연도, 장르, 플랫폼)를 설정하면 대시보드가 실시간으로 업데이트됩니다.
        - `Data` 탭에서 필터된 결과를 CSV로 내려받을 수 있습니다.
        """
    )

st.caption('Made with ❤️ — Streamlit')
