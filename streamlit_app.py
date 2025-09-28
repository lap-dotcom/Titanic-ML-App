import os
import pickle
import pandas as pd
import streamlit as st


def get_user_data() -> pd.DataFrame:
    user_data = {}

    user_data['age'] = st.slider('Age:', 0, 100, 20, 1)
    user_data['fare'] = st.slider('How much did your ticket cost you?:', 0, 300, 80, 1)
    user_data['sibsp'] = st.slider('Number of siblings and spouses aboard:', 0, 15, 3, 1)
    user_data['parch'] = st.slider('Number of parents and children aboard:', 0, 15, 3, 1)

    col1, col2, col3 = st.columns(3)
    user_data['pclass'] = col1.radio('Ticket class:', ['1st', '2nd', '3rd'])
    user_data['sex'] = col2.radio('Sex:', ['Man', 'Woman'])
    user_data['embarked'] = col3.radio('Port of Embarkation:', ['Cherbourg', 'Queenstown', 'Southampton'], index=1)

    for k in user_data.keys():
        user_data[k] = [user_data[k]]
    df = pd.DataFrame(user_data)

    df['sex'] = df['sex'].map({'Man': 'male', 'Woman': 'female'})
    df['pclass'] = df['pclass'].map({'1st': 1, '2nd': 2, '3rd': 3})
    df['embarked'] = df['embarked'].map({'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'})
    df['num_relatives'] = df['sibsp'] + df['parch']

    return df


@st.cache_resource
def load_model(model_file_path: str):
    with st.spinner("Loading model..."):
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
    return model


def main():
    model_name = 'trained_grad_boost.pkl'
    this_file_path = os.path.abspath(__file__)
    project_path = '/'.join(this_file_path.split('/')[:-2])

    st.header('Would you have survived the Titanic?')
    df_user_data = get_user_data()

    st.write("Your data:")
    st.write(df_user_data)

    st.image(project_path + '/images/RMS_Titanic.jpg')


if __name__ == '__main__':
    main()
