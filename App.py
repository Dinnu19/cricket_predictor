import streamlit as st
from streamlit_option_menu import option_menu
import xgboost
from xgboost import XGBRegressor
import pickle
import pandas as pd
# with st.sidebar:
#     selected=option_menu(
#         menu_title=None,
#         options=["Ipl win predictor","T20 Score predictor"],
#         orientation="horizontal",
#         default_index=0
#     )
selected=option_menu(
        menu_title=None,
        options=["Ipl win predictor","T20 Score predictor"],
        orientation="horizontal",
        default_index=0
    )
if(selected=="T20 Score predictor"):
    pipe = pickle.load(open('pipe1.pkl','rb'))

    teams = ['Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'West Indies',
    'Afghanistan',
    'Pakistan',
    'Sri Lanka']

    cities = ['Colombo', 'Durban', 'Wellington', 'Auckland', 'Pallekele',
       'Lauderhill', 'Cape Town', 'Southampton', 'Johannesburg', 'Mirpur',
       'St Lucia', 'Dubai', 'London', 'Manchester', 'Mumbai', 'Barbados',
       'Lahore', 'Nottingham']

    st.title('T20 International Score Predictor')

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Select batting team',sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select bowling team', sorted(teams))

    city = st.selectbox('Select city',sorted(cities))

    col3,col4,col5 = st.columns(3)

    with col3:
        current_score = st.number_input('Current Score')
    with col4:
        overs = st.number_input('Overs done(works for over>5)')
    with col5:
        wickets = st.number_input('Wickets out')

    last_five = st.number_input('Runs scored in last 5 overs')

    if st.button('Predict Score'):
        balls_left = 120 - (overs*6)
        wickets_left = 10 -wickets
        crr = current_score/overs

        input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
        result = pipe.predict(input_df)
        st.header("Predicted Score - " + str(int(result[0])))


if(selected=="Ipl win predictor"):
        teams = ['Sunrisers Hyderabad',
        'Mumbai Indians',
        'Royal Challengers Bangalore',
        'Kolkata Knight Riders',
        'Kings XI Punjab',
        'Chennai Super Kings',
        'Rajasthan Royals',
        'Delhi Capitals']

        cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
        'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
        'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
        'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
        'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
        'Sharjah', 'Mohali', 'Bengaluru']

        pipe = pickle.load(open('pipe.pkl','rb'))
        st.title('IPL Win Predictor')

        col1, col2 = st.columns(2)

        with col1:
           batting_team = st.selectbox('Select the batting team',sorted(teams))
        with col2:
           bowling_team = st.selectbox('Select the bowling team',sorted(teams))

        selected_city = st.selectbox('Select host city',sorted(cities))

        target = st.number_input('Target')

        col3,col4,col5 = st.columns(3)

        with col3:
            score = st.number_input('Score')
        with col4:
            overs = st.number_input('Overs completed')
        with col5:
            wickets = st.number_input('Wickets out')

        if st.button('Predict Probability'):
            runs_left = target - score
            balls_left = 120 - (overs*6)
            wickets = 10 - wickets
            crr = score/overs
            rrr = (runs_left*6)/balls_left

            input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets],'total_runs_x':[target],'current_run_rate':[crr],'required_run_rate':[rrr]})

            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]
            st.header(batting_team + "- " + str(round(win*100)) + "%")
            st.header(bowling_team + "- " + str(round(loss*100)) + "%")