import os
import pickle

import pandas as pd
import streamlit as st

# https://docs.streamlit.io/library/api-reference

# HWO TO RUN THE APP:
# streamlit run <YOUR_PATH>/titanic_streamlit/main_app/streamlit_app.py


def get_user_data() -> pd.DataFrame:
    """
    Get the data provided by the user. Preprocess the data and create a
    DataFrame to feed the model and make the prediction.

    :return: preprocessed user information from the app
    """
    user_data = {}

    user_data['age'] = st.slider(
        label='Age:',
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

    user_data['fare'] = st.slider(
        label='How much did your ticket cost you?:',
        min_value=0,
        max_value=300,
        value=80,
        step=1
    )

    user_data['sibsp'] = st.slider(
        label='Number of siblings and spouses aboard:',
        min_value=0,
        max_value=15,
        value=3,
        step=1
    )

    user_data['parch'] = st.slider(
        label='Number of parents and children aboard:',
        min_value=0,
        max_value=15,
        value=3,
        step=1
    )

    col1, col2, col3 = st.columns(3)

    user_data['pclass'] = col1.radio(
        label='Ticket class:',
        options=['1st', '2nd', '3rd'],
        horizontal=False
    )

    user_data['sex'] = col2.radio(
        label='Sex:',
        options=['Man', 'Woman'],
        horizontal=False
    )

    user_data['embarked'] = col3.radio(
        label='Port of Embarkation:',  # hidden
        options=['Cherbourg', 'Queenstown', 'Southampton'],
        index=1
    )

    # turn dict 'values' to list before turning the dict into a DataFrame
    for k in user_data.keys():
        user_data[k] = [user_data[k]]
    df = pd.DataFrame(data=user_data)

    # some preprocessing of the raw data from the user.
    # Follow the same data structure than in the Kaggle competition
    df['sex'] = df['sex'].map({'Man': 'male', 'Woman': 'female'})
    df['pclass'] = df['pclass'].map({'1st': 1, '2nd': 2, '3rd': 3})
    df['embarked'] = df['embarked'].map(
        {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}
    )
    df['num_relatives'] = df['sibsp'] + df['parch']

    return df


@st.cache_resource
def load_model(model_file_path: str):
    """
    Load a model in pickle format (.pkl extension) from the '/models' directory

    :param model_file_path: where the trained model is stored in pickle format
    :return: the trained model, a sklearn object
    """

    with st.spinner("Loading model..."):
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)

    return model


def main():
    # choose the trained model you want to use to make predictions
    model_name = 'trained_grad_boost.pkl'

    # get the project file name: "<your_project_path>/titanic_streamlit"
    this_file_path = os.path.abspath(__file__)
    project_path = '/'.join(this_file_path.split('/')[:-2])

    # title
    st.header(body='Would you have survived the Titanic?🚢')

    # get the data from the user
    df_user_data = get_user_data()

    # load the model and predict the outcome for the given user data
    model = load_model(model_file_path=project_path + '/models/' + model_name)
    prob = model.predict_proba(df_user_data)[0][1]
    prob = int(prob * 100)

    emojis = ["😕", "🙃", "🙂" , "😀"]
    state = min(prob // 25, 3)  # [0,1,2,3] ~= [horrible, bad, good, great]

    st.write('')
    st.title(f'{prob}% chance to survive! {emojis[state]}')
    if state == 0:  # 0-24% chance to survive
        st.error(
            "Bad news my friend, you will be food for sharks! 🦈"
        )
    elif state == 1:  # 25-49% chance to survive
        st.warning(
            "Hey... I hope you know how to swim, maybe you have to do it! 🏊‍♂️"
        )
    elif state == 2:  # 50-74% chance to survive
        st.info(
            "Well done! You are on the right track, but don't get lost! 😙"
        )
    else:  # 75-100% chance to survive
        st.success(
            'Congratulations! You can rest assured, you will be fine! 🤩'
        )

    # display an image of the Titanic
    st.image(project_path + '/images/RMS_Titanic.jpg')


if __name__ == '__main__':
    main()
