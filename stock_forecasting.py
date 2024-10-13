import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Custom CSS to improve the look
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stPlotlyChart {
        height: 500px;
    }
    .stDataFrame {
        height: 400px;
        overflow: auto;
    }
    h1, h2 {
        color: #3366cc;
    }
    .stRadio > label {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        width: 100%;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# List of popular US stocks
popular_tickers = {
    "TSLA": "Tesla Inc.",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc. (Google)",
    "META": "Meta Platforms Inc. (Facebook)",
    "TSLA": "Tesla Inc.",
    "BRK.A": "Berkshire Hathaway Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson",
    "V": "Visa Inc.",
    "PG": "Procter & Gamble Company",
    "UNH": "UnitedHealth Group Incorporated",
    "MA": "Mastercard Incorporated",
    "NVDA": "NVIDIA Corporation",
    "HD": "The Home Depot Inc.",
    "DIS": "The Walt Disney Company",
    "BAC": "Bank of America Corporation",
    "ADBE": "Adobe Inc.",
    "NFLX": "Netflix Inc.",
    "XOM": "Exxon Mobil Corporation"
}

# Utility functions
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df.index = df.index.tz_localize(None)
    return df

def calculate_moving_averages(df, short_window, long_window):
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    return df

def generate_signals(df):
    df['Signal'] = 0
    df.loc[df['Short_MA'] > df['Long_MA'], 'Signal'] = 1
    df['Position'] = df['Signal'].diff()
    return df

def backtest_strategy(df):
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def plot_backtest_results(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy_Returns'], mode='lines', name='Moving Average Strategy'))
    fig.update_layout(title=f'Backtesting Results: {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Returns',
                      legend_title='Strategy')
    return fig

def run_forecast(ticker, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    df = fetch_stock_data(ticker, start_date, end_date)

    prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
    fig.update_layout(title=f'Stock Price Forecast: {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      legend_title='Data')

    return fig, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)

@st.cache_data
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    return stock.info

def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_volume(df):
    fig = px.bar(df, x=df.index, y='Volume', title='Trading Volume')
    return fig

def fetch_comparison_data(ticker, start_date, end_date):
    indices = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ'}
    
    # Ensure start_date and end_date are timezone-naive
    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)
    
    # Download data
    tickers = [ticker] + list(indices.keys())
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Rename columns
    data.columns = [ticker] + list(indices.values())
    
    # Ensure all data is timezone-naive
    data.index = data.index.tz_localize(None)
    
    # Calculate cumulative returns
    return data.pct_change().fillna(0).cumsum()

def plot_comparison(df, title):
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Cumulative Returns', legend_title='Ticker')
    return fig

def safe_percentage(value):
    if value is None:
        return 'N/A'
    try:
        return f"{float(value):.2%}"
    except (ValueError, TypeError):
        return str(value)

def fetch_nasdaq_100():
    nasdaq_100 = yf.Ticker('^NDX').info.get('components', [])
    return sorted(nasdaq_100)

def fetch_sp_500():
    sp_500 = yf.Ticker('^GSPC').info.get('components', [])
    return sorted(sp_500)

def create_merged_signal_visualization(signal_summary):
    # Create a single row for the merged visualization
    fig = go.Figure()

    # Define colors for each signal
    colors = {'Buy': 'green', 'Sell': 'red', 'Hold': 'yellow'}

    # Add a bar for each signal
    for signal, count in signal_summary.items():
        fig.add_trace(go.Bar(
            x=[signal],
            y=[count],
            name=signal,
            marker_color=colors[signal],
            text=[f"{signal}: {count}"],
            textposition='auto'
        ))

    # Add traffic light indicators
    for i, (signal, count) in enumerate(signal_summary.items()):
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=count,
            title={"text": signal},
            domain={'x': [i/3, (i+1)/3], 'y': [0, 0.3]},
            delta={'reference': 0, 'relative': False},
            number={'font': {'color': colors[signal]}},
        ))

    # Update layout
    fig.update_layout(
        title="Merged Signal Summary and Traffic Lights",
        xaxis_title="Signals",
        yaxis_title="Count",
        barmode='group',
        height=500,
    )

    return fig

# Main app
def main():
    st.title('📊 Stock Analysis Dashboard')

    st.subheader('Stock Selection')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        custom_ticker = st.text_input('Enter a custom stock ticker:')

    with col2:
        popular_option = st.selectbox(
            'Choose a popular stock:',
            [''] + list(popular_tickers.keys()),
            format_func=lambda x: f"{x} - {popular_tickers.get(x, '')}" if x else "Select a popular stock"
        )

    with col3:
        nasdaq_100 = fetch_nasdaq_100()
        nasdaq_option = st.selectbox(
            'Choose from NASDAQ 100:',
            [''] + ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'AVGO', 'ADBE', 
                    'COST', 'PEP', 'CSCO', 'TMUS', 'CMCSA', 'TXN', 'NFLX', 'QCOM', 'HON', 'INTC', 
                    'INTU', 'AMD', 'AMGN', 'AMAT', 'BKNG', 'SBUX', 'ISRG', 'MDLZ', 'ADI', 'PYPL', 
                    'REGN', 'VRTX', 'GILD', 'ADP', 'MRNA', 'LRCX', 'PANW', 'MU', 'CHTR', 'KLAC', 
                    'SNPS', 'CDNS', 'MELI', 'ASML', 'ABNB', 'ORLY', 'ATVI', 'WDAY', 'FTNT', 'KDP', 
                    'MNST', 'KHC', 'ADSK', 'NXPI', 'MAR', 'CTAS', 'MCHP', 'PAYX', 'PCAR', 'LULU', 
                    'DXCM', 'IDXX', 'MRVL', 'AEP', 'ODFL', 'EXC', 'BIIB', 'CPRT', 'ROST', 'SIRI', 
                    'DLTR', 'EA', 'XEL', 'CTSH', 'WBA', 'VRSK', 'FAST', 'CSGP', 'ILMN', 'EBAY', 
                    'ZS', 'ANSS', 'DDOG', 'FANG', 'TEAM', 'ALGN', 'CRWD', 'ENPH', 'SGEN', 'CEG', 
                    'BKR', 'MTCH', 'SPLK', 'LCID', 'ZM', 'RIVN', 'GEHC', 'DASH', 'TTWO', 'VRSN', 
                    'SWKS', 'DOCU', 'OKTA'],
            format_func=lambda x: x if x else "Select from NASDAQ 100"
        )

    with col4:
        sp_500 = fetch_sp_500()
        sp_option = st.selectbox(
            'Choose from S&P 500:',
            [''] + ['A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF.B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLL', 'BMY', 'BR', 'BRK.B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CERN', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CRL', 'CRM', 'CSCO', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CTXS', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH', 'EOG', 'EQIX', 'EQR', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FB', 'FBHS', 'FCX', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GPS', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HBI', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INFO', 'INTC', 'INTU', 'IP', 'IPG', 'IPGP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KSU', 'L', 'LDOS', 'LEG', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUMN', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NLOK', 'NLSN', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PAYC', 'PAYX', 'PBCT', 'PCAR', 'PEAK', 'PEG', 'PENN', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PTC', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TWTR', 'TXN', 'TXT', 'TYL', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VIAC', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WLTW', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WU', 'WY', 'WYNN', 'XEL', 'XLNX', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS'],
            format_func=lambda x: x if x else "Select from S&P 500"
        )

    # Determine which ticker to use
    ticker = custom_ticker or popular_option or nasdaq_option or sp_option

    if not ticker:
        st.warning("Please select a stock or enter a custom ticker.")
        st.stop()

    st.markdown(f"<h2>**Selected stock: {ticker}**</h2>", unsafe_allow_html=True)
    st.markdown("---")  # Add a horizontal line for emphasis

    info = get_stock_info(ticker)

    st.write(f"**Selected Stock: {info.get('longName', 'N/A')}** ({ticker})")
    st.write(f"Current Price: ${info.get('currentPrice', 'N/A')} | 52 Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")
    
    # Create columns for current price and 52-week range chart
    price_col, chart_col = st.columns([1, 3])

    with price_col:
        st.write(f"Current Price: ${info.get('currentPrice', 'N/A')}")
        st.write(f"52 Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")

        # Add real-time stock price
        real_time_price = yf.Ticker(ticker).info.get('currentPrice', 'N/A')
        # Check if real_time_price is available and valid
        if real_time_price != 'N/A' and real_time_price is not None:
            st.write(f"Real-time Price: ${real_time_price:.2f}")
        else:
            st.write("Real-time Price: Not available")
            st.write("Possible reasons: Market closed, API issue, or invalid ticker symbol")

    with chart_col:
        # Create a line chart with 52-week range and current price
        fig_range = go.Figure()

        # Add the 52-week range line
        fig_range.add_trace(go.Scatter(
            x=[info.get('fiftyTwoWeekLow', 0), info.get('fiftyTwoWeekHigh', 0)],
            y=['Price Range', 'Price Range'],
            mode='lines',
            line=dict(color='lightgrey', width=4),
            name='52 Week Range'
        ))

        # Add the current price marker
        fig_range.add_trace(go.Scatter(
            x=[info.get('currentPrice', 0)],
            y=['Price Range'],
            mode='markers',
            marker=dict(size=20, color='red', symbol='diamond'),
            name='Current Price'
        ))

        # Update layout
        fig_range.update_layout(
            title='52 Week Price Range with Current Price',
            xaxis_title='Price ($)',
            showlegend=True,
            height=200,
            width=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        # Add annotations for min, max, and current price
        fig_range.add_annotation(x=info.get('fiftyTwoWeekLow', 0), y=0,
                                 text=f"Low: ${info.get('fiftyTwoWeekLow', 0):.2f}", 
                                 showarrow=False, yshift=-30)
        fig_range.add_annotation(x=info.get('fiftyTwoWeekHigh', 0), y=0,
                                 text=f"High: ${info.get('fiftyTwoWeekHigh', 0):.2f}", 
                                 showarrow=False, yshift=-30)
        fig_range.add_annotation(x=info.get('currentPrice', 0), y=0,
                                 text=f"Current: ${info.get('currentPrice', 0):.2f}", 
                                 showarrow=False, yshift=30, font=dict(size=14, color='red'))

        # Display the chart
        st.plotly_chart(fig_range, use_container_width=True, config={'displayModeBar': False})
        fig_range.update_layout(margin=dict(b=0))
        
    # Create two columns for time range selection and plot
    time_range_col, plot_col = st.columns([1, 3])

    with time_range_col:
        # Add time range selection
        time_ranges = ['max', '10y', '5y', '3y', '1y', '6mo', '3mo', '1mo', '5d', '1d']
        st.write("Select Time Range:")
        selected_range = None
        col1, col2 = st.columns(2)
        for i, range in enumerate(time_ranges):
            if i < len(time_ranges) // 2:
                if col1.button(range, key=f'btn_{range}_1', use_container_width=True):
                    selected_range = range
            else:
                if col2.button(range, key=f'btn_{range}_2', use_container_width=True):
                    selected_range = range
        if selected_range is None:
            selected_range = 'max'  # Default selection

    with plot_col:
        # Fetch historical data based on selected range
        hist_data = yf.Ticker(ticker).history(period=selected_range)

        # Create a candlestick chart with line for historical prices
        fig_hist = go.Figure()

        # Add candlestick trace
        fig_hist.add_trace(go.Candlestick(x=hist_data.index,
                                          open=hist_data['Open'],
                                          high=hist_data['High'],
                                          low=hist_data['Low'],
                                          close=hist_data['Close'],
                                          name='Candlestick'))

        # Add line trace
        fig_hist.add_trace(go.Scatter(x=hist_data.index, 
                                      y=hist_data['Close'], 
                                      mode='lines', 
                                      name='Close Price',
                                      line=dict(color='blue', width=1)))

        # Update layout
        fig_hist.update_layout(title=f'{ticker} Stock Price - {selected_range}', 
                               xaxis_title='Date', 
                               yaxis_title='Price ($)',
                               xaxis_rangeslider_visible=False,
                               yaxis2=dict(
                                   title='Percent Change (%)',
                                   overlaying='y',
                                   side='right',
                                   ticklen=0,
                                   tickcolor='rgba(0,0,0,0)'
                               ))

        # Display the chart
        st.plotly_chart(fig_hist, use_container_width=True)

    main_col1, main_col2 = st.columns([1, 3])

    with main_col1:
        st.header('Analysis Options')
        page = st.radio("Select Analysis", 
                        ["Stock Info", "Stock Performance", "Stock Backtesting", "Forecasting", "Technical Analysis"])

        st.subheader('Company Info')
        st.write(f"Sector: {info.get('sector', 'N/A')}")
        st.write(f"Industry: {info.get('industry', 'N/A')}")
        market_cap = info.get('marketCap', 'N/A')
        if isinstance(market_cap, (int, float)):
            st.write(f"Market Cap: ${market_cap:,.2f}")
        else:
            st.write(f"Market Cap: {market_cap}")
        try:
            st.write(f"P/E Ratio: {info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "P/E Ratio: N/A")
        except Exception as e:
            st.write(f"P/E Ratio: Error - {str(e)}")

        try:
            st.write(f"Dividend Yield: {safe_percentage(info.get('dividendYield', 'N/A'))}")
        except Exception as e:
            st.write(f"Dividend Yield: Error - {str(e)}")

        try:
            st.write(f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if isinstance(info.get('fiftyTwoWeekHigh'), (int, float)) else "52 Week High: N/A")
        except Exception as e:
            st.write(f"52 Week High: Error - {str(e)}")

        try:
            st.write(f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if isinstance(info.get('fiftyTwoWeekLow'), (int, float)) else "52 Week Low: N/A")
        except Exception as e:
            st.write(f"52 Week Low: Error - {str(e)}")

        try:
            st.write(f"Beta: {info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else "Beta: N/A")
        except Exception as e:
            st.write(f"Beta: Error - {str(e)}")

        try:
            st.write(f"EPS (TTM): ${info.get('trailingEps', 'N/A'):.2f}" if isinstance(info.get('trailingEps'), (int, float)) else "EPS (TTM): N/A")
        except Exception as e:
            st.write(f"EPS (TTM): Error - {str(e)}")

        try:
            st.write(f"Price to Book: {info.get('priceToBook', 'N/A'):.2f}" if isinstance(info.get('priceToBook'), (int, float)) else "Price to Book: N/A")
        except Exception as e:
            st.write(f"Price to Book: Error - {str(e)}")

        try:
            st.write(f"Debt to Equity: {info.get('debtToEquity', 'N/A'):.2f}" if isinstance(info.get('debtToEquity'), (int, float)) else "Debt to Equity: N/A")
        except Exception as e:
            st.write(f"Debt to Equity: Error - {str(e)}")

        try:
            st.write(f"Shares Outstanding: {info.get('sharesOutstanding', 'N/A'):,}" if isinstance(info.get('sharesOutstanding'), (int, float)) else "Shares Outstanding: N/A")
        except Exception as e:
            st.write(f"Shares Outstanding: Error - {str(e)}")

        try:
            st.write(f"Number of Employees: {info.get('fullTimeEmployees', 'N/A'):,}" if isinstance(info.get('fullTimeEmployees'), (int, float)) else "Number of Employees: N/A")
        except Exception as e:
            st.write(f"Number of Employees: Error - {str(e)}")

        try:
            st.write(f"Headquarters: {info.get('city', 'N/A')}, {info.get('state', 'N/A')}, {info.get('country', 'N/A')}")
        except Exception as e:
            st.write(f"Headquarters: Error - {str(e)}")

    with main_col2:
        if page == "Stock Info":
            st.header('Stock Information')
            st.subheader("Company Description")
            st.markdown(f"**{info.get('longBusinessSummary', 'N/A')}**")
            st.markdown("---")  # Add a horizontal line for visual separation
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            # Create a bar plot for Market Cap
            fig_market_cap = px.bar(x=['Market Cap'], y=[info.get('marketCap', 0)], 
                                    labels={'x': '', 'y': 'Value ($)'}, title='Market Cap')
            fig_market_cap.update_layout(showlegend=False)
            # Calculate total market cap for S&P 500 and Nasdaq 100
            # The S&P 500 is an index, not a single stock, so it doesn't have a market cap.
            # Instead, we should calculate the total market cap of all S&P 500 components.
            try:
                aapl = yf.Ticker("AAPL")
                msft = yf.Ticker("MSFT")
                aapl_market_cap = aapl.info.get('marketCap', 0)
                msft_market_cap = msft.info.get('marketCap', 0)
                if aapl_market_cap == 0 or msft_market_cap == 0:
                    raise ValueError("Apple or Microsoft market cap calculation resulted in 0")
            except Exception as e:
                st.warning(f"Unable to fetch company data: {str(e)}. Using 0 as fallback.")
                aapl_market_cap = 0
                msft_market_cap = 0

            # Create a bar plot for Market Cap comparison
            fig_market_cap = px.bar(
                x=[info.get('shortName', 'Selected Stock'), 'Apple', 'Microsoft'],
                y=[info.get('marketCap', 0), aapl_market_cap, msft_market_cap],
                labels=dict(x='', y='Market Cap ($)', color='Company'),
                title='Market Cap Comparison',
                color=['Selected Stock', 'Apple', 'Microsoft'],
                text_auto=True
            )
            fig_market_cap.update_traces(texttemplate='%{y:.2s}', textposition='outside')
            fig_market_cap.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig_market_cap.update_xaxes(showticklabels=False)
            fig_market_cap.update_layout(coloraxis_colorbar=dict(title='Market Cap ($)'))
            fig_market_cap.update_layout(showlegend=True)
            
            metrics_col1.plotly_chart(fig_market_cap, use_container_width=True)
            metrics_col1.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")

            # Create a gauge plot for P/E Ratio
            fig_pe_ratio = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = info.get('trailingPE', 0),
                title = {'text': "P/E Ratio"},
                gauge = {'axis': {'range': [None, 50]}}
            ))
            metrics_col2.plotly_chart(fig_pe_ratio, use_container_width=True)
            try:
                trailing_pe = info.get('trailingPE', 'N/A')
                if trailing_pe != 'N/A':
                    metrics_col2.metric("P/E Ratio", f"{trailing_pe:.2f}")
                else:
                    metrics_col2.metric("P/E Ratio", "N/A")
            except Exception as e:
                st.warning(f"Error displaying P/E Ratio: {str(e)}")
                metrics_col2.metric("P/E Ratio", "Error")

            # Create a pie chart for Dividend Yield
            # Fetch dividend yield for Apple and Microsoft
            try:
                aapl = yf.Ticker("AAPL")
                msft = yf.Ticker("MSFT")
                aapl_dividend_yield = aapl.info.get('dividendYield', 0)
                msft_dividend_yield = msft.info.get('dividendYield', 0)
            except Exception as e:
                st.warning(f"Unable to fetch dividend data: {str(e)}. Using 0 as fallback.")
                aapl_dividend_yield = 0
                msft_dividend_yield = 0

            # Create a bar plot for Dividend Yield comparison
            dividend_yield = info.get('dividendYield', 0)
            fig_dividend = px.bar(
                x=['Selected Stock', 'Apple', 'Microsoft'],
                y=[dividend_yield, aapl_dividend_yield, msft_dividend_yield],
                labels=dict(x='', y='Dividend Yield', color='Company'),
                title='Dividend Yield Comparison',
                color=['Selected Stock', 'Apple', 'Microsoft'],
                text_auto=True
            )
            fig_dividend.update_traces(texttemplate='%{y:.2%}', textposition='outside')
            fig_dividend.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig_dividend.update_xaxes(showticklabels=True)
            fig_dividend.update_layout(showlegend=True)
            
            metrics_col3.plotly_chart(fig_dividend, use_container_width=True)
            metrics_col3.metric("Dividend Yield", safe_percentage(dividend_yield))

            # Add a new row for time plots
            st.subheader('Historical Trends')
            trend_col1, trend_col2, trend_col3 = st.columns(3)

            # Fetch historical data
            historical_data = fetch_stock_data(ticker, datetime.now() - timedelta(days=365*5), datetime.now())

            # Market Cap trend
            fig_market_cap_trend = px.line(historical_data, x=historical_data.index, y=historical_data['Close'] * historical_data['Volume'],
                                           title='Market Cap Trend', labels={'y': 'Market Cap', 'x': 'Date'})
            trend_col1.plotly_chart(fig_market_cap_trend, use_container_width=True)

            # P/E Ratio trend (using a simple calculation, might not be entirely accurate)
            try:
                trailing_eps = info.get('trailingEps', None)
                if trailing_eps is not None and trailing_eps != 0:
                    historical_data['P/E Ratio'] = historical_data['Close'] / trailing_eps
                    fig_pe_trend = px.line(historical_data, x=historical_data.index, y='P/E Ratio',
                                           title='P/E Ratio Trend', labels={'y': 'P/E Ratio', 'x': 'Date'})
                    trend_col2.plotly_chart(fig_pe_trend, use_container_width=True)
                else:
                    trend_col2.warning("Unable to calculate P/E Ratio: Trailing EPS is zero or not available.")
            except Exception as e:
                trend_col2.error(f"An error occurred while calculating P/E Ratio trend: {str(e)}")

            # Dividend Yield trend (if available)
            if 'Dividends' in historical_data.columns:
                historical_data['Dividend Yield'] = historical_data['Dividends'] / historical_data['Close'] * 100
                fig_dividend_trend = px.line(historical_data, x=historical_data.index, y='Dividend Yield',
                                             title='Dividend Yield Trend', labels={'y': 'Dividend Yield (%)', 'x': 'Date'})
                trend_col3.plotly_chart(fig_dividend_trend, use_container_width=True)
            else:
                trend_col3.write("Dividend data not available for this stock.")
                
            # Add latest news about the company
            st.subheader("Latest News")
            try:
                news = yf.Ticker(ticker).news
                if news:
                    for i, article in enumerate(news[:5]):  # Display up to 5 latest news articles
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if 'thumbnail' in article and article['thumbnail']:
                                st.image(article['thumbnail']['resolutions'][0]['url'], width=100)
                            else:
                                st.image("https://via.placeholder.com/100x100.png?text=No+Image", width=100)
                        with col2:
                            st.write(f"**{article['title']}**")
                            st.write(f"*{article['publisher']}* - {article['providerPublishTime']}")
                            st.write(article['link'])
                        if i < 4:  # Add a separator between articles, except after the last one
                            st.markdown("---")
                else:
                    st.write("No recent news available for this company.")
            except Exception as e:
                st.write(f"Unable to fetch news: {str(e)}")

        elif page == "Stock Performance":
            st.header('Stock Performance')
            
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input('Start Date', datetime.now() - timedelta(days=365))
            with date_col2:
                end_date = st.date_input('End Date', datetime.now())

            if start_date >= end_date:
                st.error("Error: Start date must be before end date.")
                return

            try:
                # Ensure start_date and end_date are timezone-aware
                start_date = pd.Timestamp(start_date).tz_localize('UTC')
                end_date = pd.Timestamp(end_date).tz_localize('UTC')

                df = fetch_stock_data(ticker, start_date, end_date)
                if df.empty:
                    st.warning(f"No data available for {ticker} in the selected date range. Please adjust the dates.")
                    return

                st.subheader('Price Chart')
                # Calculate RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))

                # Create subplot with shared x-axis
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

                # Add candlestick trace to first subplot
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

                # Add RSI trace to second subplot
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)

                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                # Update layout
                fig.update_layout(
                    title='Price and RSI Chart',
                    yaxis_title='Price',
                    yaxis2_title='RSI',
                    xaxis_rangeslider_visible=False,
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                st.subheader('Volume Chart')
                fig = go.Figure(data=[go.Bar(x=df.index, y=df['Volume'])])
                fig.update_layout(title='Trading Volume', xaxis_title='Date', yaxis_title='Volume')
                st.plotly_chart(fig, use_container_width=True)

                st.subheader('Performance Comparison with Major Indices')
                comparison_data = pd.DataFrame()  # Initialize as empty DataFrame
                try:
                    comparison_data = fetch_comparison_data(ticker, start_date, end_date)
                except Exception as e:
                    st.warning(f"Error fetching comparison data: {str(e)}")
                    st.info("This error might be due to timezone inconsistencies in the data. Try the following:")
                    st.info("1. Ensure all date inputs are in the same timezone.")
                    st.info("2. If using custom date ranges, try selecting preset ranges (e.g., '1 Month', '3 Months', '1 Year').")
                    st.info("3. If the issue persists, try refreshing the page or selecting a different date range.")

                if comparison_data.empty:
                    st.warning("No comparison data available. This could be due to several reasons:")
                    st.info("1. The selected date range might be too short or recent.")
                    st.info("2. There might be an issue with the data source.")
                    st.info("3. The stock ticker might not have data for the selected period.")
                    st.info("Please try adjusting the date range or selecting a different stock.")
                else:
                    st.plotly_chart(plot_comparison(comparison_data, f'{ticker} vs Major Indices'), use_container_width=True)

                sector = info.get('sector', '')
                if sector and not comparison_data.empty:
                    sector_etfs = {
                        'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
                        'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP',
                        'Energy': 'XLE', 'Materials': 'XLB', 'Industrials': 'XLI',
                        'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Communication Services': 'XLC'
                    }
                    if sector in sector_etfs:
                        st.subheader(f'Performance Comparison with {sector} Sector ETF')
                        sector_etf = sector_etfs[sector]
                        sector_comparison = pd.DataFrame()  # Initialize as empty DataFrame
                        try:
                            sector_comparison = fetch_comparison_data(f"{ticker} {sector_etf}", start_date, end_date)
                        except Exception as e:
                            st.warning(f"Error fetching sector comparison data: {str(e)}")

                        if not sector_comparison.empty:
                            st.plotly_chart(plot_comparison(sector_comparison, f'{ticker} vs {sector} Sector ETF ({sector_etf})'), use_container_width=True)
                        else:
                            st.warning(f"No sector comparison data available for {sector} in the selected date range.")

                # Continue with the rest of your analysis here...

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try the following:")
                st.info("1. Adjust the date range to a more recent period.")
                st.info("2. Check if the selected stock ticker is valid.")
                st.info("3. Ensure you have a stable internet connection.")
                st.info("If the problem persists, the data source might be temporarily unavailable.")

        elif page == "Stock Backtesting":
            st.header('Stock Backtesting')
            
            # Add your backtesting parameters here
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input('Start Date', datetime.now() - timedelta(days=365*5))
            with col2:
                end_date = st.date_input('End Date', datetime.now())
            
            # Add strategy parameters
            st.subheader('Strategy Parameters')
            col1, col2 = st.columns(2)
            with col1:
                short_window = st.slider('Short-term MA window', 10, 100, 50)
            with col2:
                long_window = st.slider('Long-term MA window', 50, 300, 200)

            if st.button('Run Backtest'):
                try:
                    # Fetch data
                    df = fetch_stock_data(ticker, start_date, end_date)
                    
                    # Perform backtesting
                    df = calculate_moving_averages(df, short_window, long_window)
                    df = generate_signals(df)
                    df = backtest_strategy(df)

                    # Plot results
                    fig = plot_backtest_results(df, ticker)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display metrics
                    buy_hold_return = df['Cumulative_Returns'].iloc[-1]
                    strategy_return = df['Cumulative_Strategy_Returns'].iloc[-1]

                    metrics_col1, metrics_col2 = st.columns(2)
                    metrics_col1.metric("Buy and Hold Return", f"{buy_hold_return:.2f}x")
                    metrics_col2.metric("Strategy Return", f"{strategy_return:.2f}x")

                    # Display trade signals
                    st.subheader(f'Trade Signals for {ticker}')
                    signals = df[df['Position'] != 0].copy()
                    signals['Action'] = signals['Position'].map({1: 'Buy', -1: 'Sell'})
                    signals = signals[['Close', 'Action', 'Short_MA', 'Long_MA']]
                    signals = signals.reset_index()
                    signals.columns = ['Date', 'Close', 'Action', 'Short_MA', 'Long_MA']
                    signals['Date'] = signals['Date'].dt.strftime('%Y-%m-%d')
                    signals = signals.set_index('Date')
                    
                    # Create two columns
                    col1, col2 = st.columns(2)
                    
                    # Display the trade signals dataframe
                    with col1:
                        st.subheader("Trade Signals")
                        
                        # Calculate additional signals
                        # Define the calculate_rsi function
                        def calculate_rsi(prices, window=14):
                            delta = prices.diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            return rsi
                        
                        signals['RSI'] = calculate_rsi(df['Close'])
                        # Define the calculate_macd function
                        def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
                            short_ema = prices.ewm(span=short_window, adjust=False).mean()
                            long_ema = prices.ewm(span=long_window, adjust=False).mean()
                            macd = short_ema - long_ema
                            signal_line = macd.ewm(span=signal_window, adjust=False).mean()
                            return macd, signal_line

                        signals['MACD'], signals['Signal_Line'] = calculate_macd(df['Close'])
                        # Calculate Bollinger Bands
                        def calculate_bollinger_bands(prices, window=20, num_std=2):
                            rolling_mean = prices.rolling(window=window).mean()
                            rolling_std = prices.rolling(window=window).std()
                            upper_band = rolling_mean + (rolling_std * num_std)
                            lower_band = rolling_mean - (rolling_std * num_std)
                            return upper_band, lower_band

                        signals['Bollinger_Upper'], signals['Bollinger_Lower'] = calculate_bollinger_bands(df['Close'])
                        
                        # Add signal interpretations
                        import numpy as np

                        signals['RSI_Signal'] = np.where(signals['RSI'] > 70, 'Overbought', np.where(signals['RSI'] < 30, 'Oversold', 'Neutral'))
                        signals['MACD_Signal'] = np.where(signals['MACD'] > signals['Signal_Line'], 'Bullish', 'Bearish')
                        signals['Bollinger_Signal'] = np.where(signals['Close'] > signals['Bollinger_Upper'], 'Overbought', 
                                                       np.where(signals['Close'] < signals['Bollinger_Lower'], 'Oversold', 'Neutral'))
                        
                        # Display the enhanced trade signals dataframe
                        st.dataframe(signals.style.format({
                            'Close': '${:.2f}',
                            'Short_MA': '${:.2f}',
                            'Long_MA': '${:.2f}',
                            'RSI': '{:.2f}',
                            'MACD': '{:.2f}',
                            'Signal_Line': '{:.2f}',
                            'Bollinger_Upper': '${:.2f}',
                            'Bollinger_Lower': '${:.2f}'
                        }).set_properties(**{'text-align': 'center'}), use_container_width=True)
                        
                        # Display summary of signals
                        st.subheader("Signal Summary")
                        st.markdown(f"""
                        <style>
                        .signal-container {{
                            display: flex;
                            justify-content: space-around;
                            align-items: stretch;
                            width: 100%;
                            margin-top: 20px;
                        }}
                        .signal-box {{
                            width: 30%;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            text-align: center;
                            transition: transform 0.3s ease;
                            display: flex;
                            flex-direction: column;
                            justify-content: space-between;
                        }}
                        .signal-box:hover {{
                            transform: translateY(-5px);
                        }}
                        .signal-title {{
                            font-size: 18px;
                            font-weight: bold;
                            margin-bottom: 10px;
                        }}
                        .signal-value {{
                            font-size: 24px;
                            font-weight: bold;
                        }}
                        </style>
                        <div class="signal-container">
                            <div class="signal-box" style="background-color: rgba(0, 0, 255, 0.1);">
                                <div class="signal-title" style="color: #3366cc;">RSI Signal</div>
                                <div class="signal-value" style="color: #00cc66;">{signals['RSI_Signal'].iloc[-1]}</div>
                            </div>
                            <div class="signal-box" style="background-color: rgba(255, 165, 0, 0.1);">
                                <div class="signal-title" style="color: #ff9900;">MACD Signal</div>
                                <div class="signal-value" style="color: #9900cc;">{signals['MACD_Signal'].iloc[-1]}</div>
                            </div>
                            <div class="signal-box" style="background-color: rgba(255, 0, 0, 0.1);">
                                <div class="signal-title" style="color: #cc3300;">Bollinger Bands Signal</div>
                                <div class="signal-value" style="color: #009999;">{signals['Bollinger_Signal'].iloc[-1]}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    # Visualize trade signals over time

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("Technical Indicators Over Time")

                    # RSI Plot
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=signals.index, y=signals['RSI'], name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(title='RSI Over Time', xaxis_title='Date', yaxis_title='RSI')
                    st.plotly_chart(fig_rsi, use_container_width=True)

                    # MACD Plot
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=signals.index, y=signals['MACD'], name='MACD'))
                    fig_macd.add_trace(go.Scatter(x=signals.index, y=signals['Signal_Line'], name='Signal Line'))
                    fig_macd.update_layout(title='MACD Over Time', xaxis_title='Date', yaxis_title='MACD')
                    st.plotly_chart(fig_macd, use_container_width=True)

                    # Bollinger Bands Plot
                    fig_bb = go.Figure()
                    fig_bb.add_trace(go.Scatter(x=signals.index, y=signals['Close'], name='Close Price'))
                    fig_bb.add_trace(go.Scatter(x=signals.index, y=signals['Bollinger_Upper'], name='Upper Band', line=dict(dash='dash')))
                    fig_bb.add_trace(go.Scatter(x=signals.index, y=signals['Bollinger_Lower'], name='Lower Band', line=dict(dash='dash')))
                    fig_bb.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig_bb, use_container_width=True)
                    
                    # Add traffic light signals based on the latest signals
                    st.subheader("Signal Traffic Lights")
                    latest_signals = signals.iloc[-1]
                    
                    def get_color(signal):
                        if signal in ['Overbought', 'Bearish']:
                            return 'red'
                        elif signal in ['Oversold', 'Bullish']:
                            return 'green'
                        else:
                            return 'yellow'
                    
                    rsi_color = get_color(latest_signals['RSI_Signal'])
                    macd_color = get_color(latest_signals['MACD_Signal'])
                    bollinger_color = get_color(latest_signals['Bollinger_Signal'])
                    
                    st.markdown(f"""
                    <style>
                    .traffic-light {{
                        width: 50px;
                        height: 50px;
                        border-radius: 50%;
                        display: inline-block;
                        margin: 0 10px;
                    }}
                    </style>
                    <div style='display: flex; justify-content: space-around; align-items: center;'>
                        <div>
                            <div class='traffic-light' style='background-color: {rsi_color};'></div>
                            <p>RSI</p>
                        </div>
                        <div>
                            <div class='traffic-light' style='background-color: {macd_color};'></div>
                            <p>MACD</p>
                        </div>
                        <div>
                            <div class='traffic-light' style='background-color: {bollinger_color};'></div>
                            <p>Bollinger</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Visualize trade signals over time
                    
                    # Add explanations for each technical indicator
                    st.subheader("Technical Indicator Explanations")
                    
                    st.markdown("""
                    ### RSI (Relative Strength Index)
                    - **Overbought (>70)**: The stock may be overvalued and could be due for a pullback.
                    - **Oversold (<30)**: The stock may be undervalued and could be due for a bounce.
                    - **Neutral (30-70)**: The stock is neither overbought nor oversold.
                    
                    ### MACD (Moving Average Convergence Divergence)
                    - **Bullish**: When the MACD line crosses above the signal line, it may indicate a good time to buy.
                    - **Bearish**: When the MACD line crosses below the signal line, it may indicate a good time to sell.
                    
                    ### Bollinger Bands
                    - **Overbought**: When the price touches or goes above the upper band, it may indicate the stock is overbought.
                    - **Oversold**: When the price touches or goes below the lower band, it may indicate the stock is oversold.
                    - **Neutral**: When the price is between the bands, it may indicate normal trading conditions.
                    
                    ### Moving Averages
                    - **Bullish**: When the short-term moving average crosses above the long-term moving average, it may indicate an uptrend.
                    - **Bearish**: When the short-term moving average crosses below the long-term moving average, it may indicate a downtrend.
                    
                    Remember, these signals should not be used in isolation. It's important to consider multiple factors and conduct thorough research before making investment decisions.
                    """)
                    
                    with col2:
                        st.subheader("Trade Signals Visualization")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=signals.index, y=signals['Close'], mode='lines', name='Stock Price'))
                        fig.add_trace(go.Scatter(x=signals[signals['Action'] == 'Buy'].index, 
                                                 y=signals[signals['Action'] == 'Buy']['Close'],
                                                 mode='markers', name='Buy Signal', marker=dict(color='green', size=10)))
                        fig.add_trace(go.Scatter(x=signals[signals['Action'] == 'Sell'].index, 
                                                 y=signals[signals['Action'] == 'Sell']['Close'],
                                                 mode='markers', name='Sell Signal', marker=dict(color='red', size=10)))
                        fig.update_layout(title='Trade Signals Over Time',
                                          xaxis_title='Date',
                                          yaxis_title='Stock Price ($)',
                                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                                          height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        

                except Exception as e:
                    st.error(f"An error occurred during backtesting: {str(e)}")
                    st.info("Please check your input parameters and try again.")

        elif page == "Forecasting":
            st.header('Stock Price Forecasting')
            
            days = st.slider('Number of days to forecast', 1, 365, 30)

            if st.button('Run Forecast'):
                with st.spinner('Generating forecast...'):
                    fig, forecast_data = run_forecast(ticker, days)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader('Forecast Data')
                    
                    # Format the date column
                    forecast_data['ds'] = pd.to_datetime(forecast_data['ds']).dt.strftime('%Y-%m-%d')
                    
                    # Round the numeric columns to 2 decimal places
                    forecast_data['yhat'] = forecast_data['yhat'].round(2)
                    forecast_data['yhat_lower'] = forecast_data['yhat_lower'].round(2)
                    forecast_data['yhat_upper'] = forecast_data['yhat_upper'].round(2)
                    
                    # Rename columns for better readability
                    forecast_data = forecast_data.rename(columns={
                        'ds': 'Date',
                        'yhat': 'Forecast',
                        'yhat_lower': 'Lower Bound',
                        'yhat_upper': 'Upper Bound'
                    })
                    
                    # Display the styled dataframe
                    st.dataframe(forecast_data.style
                                 .set_properties(**{'text-align': 'center'})
                                 .format({'Forecast': '${:.2f}', 'Lower Bound': '${:.2f}', 'Upper Bound': '${:.2f}'})
                                 .set_table_styles([
                                     {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                                     {'selector': 'td', 'props': [('text-align', 'center')]}
                                 ]),
                                 use_container_width=True)
                    
                    st.subheader('Forecasting Algorithm Description')
                    st.markdown("""
                    Our stock price forecasting algorithm uses Facebook's Prophet model, which is designed for time series forecasting with strong seasonal effects and several seasons of historical data. Here's how it works:

                    1. **Data Preparation**: We fetch historical stock price data for the selected ticker.
                    
                    2. **Prophet Model**: We use the Prophet model, which is based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
                    
                    3. **Trend Modeling**: Prophet uses a piecewise linear or logistic growth curve trend. It automatically detects changes in trends by selecting changepoints from the data.
                    
                    4. **Seasonality**: The model accounts for multiple seasonality patterns, including yearly, weekly, and daily patterns that are common in stock price data.
                    
                    5. **Holiday Effects**: While not explicitly used in this implementation, Prophet can account for holiday effects, which could be relevant for certain stocks.
                    
                    6. **Uncertainty Intervals**: The forecast includes uncertainty intervals, shown as the lower and upper bounds in the forecast data.
                    
                    7. **Future Predictions**: Based on the patterns learned from historical data, the model extrapolates these patterns to make future predictions.

                    Please note that while this model can capture many patterns in stock price data, stock prices are influenced by many external factors that cannot be predicted. Always use forecasts as one of many tools in your investment decision-making process.
                    """)

        elif page == "Technical Analysis":
            st.header('Technical Analysis')
            
            ma_col1, ma_col2 = st.columns(2)
            with ma_col1:
                short_window = st.slider('Short-term MA window', 1, 100, 50)
            with ma_col2:
                long_window = st.slider('Long-term MA window', 10, 300, 200)

            if st.button('Run Analysis'):
                with st.spinner('Performing technical analysis...'):
                    df = fetch_stock_data(ticker, datetime.now() - timedelta(days=365), datetime.now())
                    df = calculate_moving_averages(df, short_window, long_window)
                    df = generate_signals(df)
                    df = backtest_strategy(df)

                    fig = plot_backtest_results(df, ticker)
                    st.plotly_chart(fig, use_container_width=True)

                    buy_hold_return = df['Cumulative_Returns'].iloc[-1]
                    strategy_return = df['Cumulative_Strategy_Returns'].iloc[-1]

                    metrics_col1, metrics_col2 = st.columns(2)
                    metrics_col1.metric("Buy and Hold Return", f"{buy_hold_return:.2f}x")
                    metrics_col2.metric("Strategy Return", f"{strategy_return:.2f}x")

                    st.subheader('Signal Data')
                    
                    # Prepare the data
                    signal_data = df[['Close', 'Short_MA', 'Long_MA', 'Signal', 'Position', 'Strategy_Returns']].copy()
                    signal_data.index = signal_data.index.tz_localize(None)  # Remove timezone info
                    signal_data.index = signal_data.index.strftime('%Y-%m-%d')
                    signal_data = signal_data.round(2)
                    
                    # Rename columns for better readability
                    signal_data.columns = ['Close Price', 'Short-term MA', 'Long-term MA', 'Signal', 'Position Change', 'Strategy Returns']
                    
                    # Style the dataframe
                    styled_signal_data = signal_data.style\
                        .format({
                            'Close Price': '${:.2f}',
                            'Short-term MA': lambda x: '${:.2f}'.format(x) if pd.notnull(x) else 'N/A',
                            'Long-term MA': '${:.2f}',
                            'Signal': '{:.0f}',
                            'Position Change': '{:.0f}',
                            'Strategy Returns': '{:.2%}'
                        })\
                        .background_gradient(cmap='RdYlGn', subset=['Strategy Returns'])\
                        .set_properties(**{'text-align': 'center'})\
                        .set_table_styles([
                            {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                            {'selector': 'td', 'props': [('text-align', 'center')]}
                        ])
                    
                    # Display the styled dataframe
                    st.dataframe(styled_signal_data, use_container_width=True, height=400)

if __name__ == "__main__":
    main()
