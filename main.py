import speech_recognition as sr
import playsound
from IPython.core.display import display
from gtts import gTTS
import random
import time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import randomforest


class person:
    name = ''

    def setName(self, name):
        self.name = name


def there_exists(terms):
    for term in terms:
        if term in voice_data:
            return True


r = sr.Recognizer()  # initialise a recogniser


# listen for audio and convert it to text:
def record_audio(ask=False):
    with sr.Microphone() as source:  # microphone as source
        if ask:
            speak(ask)
        audio = r.listen(source)  # listen for the audio via source
        voice_data = ''
        try:
            voice_data = r.recognize_google(audio)  # convert audio to text
        except sr.UnknownValueError:  # error: recognizer does not understand
            speak('I did not get that')
        except sr.RequestError:
            speak('Sorry, the service is down')  # error: recognizer is not connected
        print(f">> {voice_data.lower()}")  # print what user said
        return voice_data.lower()


# get string and make a audio file to be played
def speak(audio_string):
    tts = gTTS(text=audio_string, lang='en')  # text to speech(voice)
    r = random.randint(1, 20000000)
    audio_file = 'audio' + str(r) + '.mp3'
    tts.save(audio_file)  # save as mp3
    playsound.playsound(audio_file)  # play the audio file
    print(f"Sovi: {audio_string}")  # print what app said
    os.remove(audio_file)  # remove audio file


def respond(voice_data):
    # 1: greeting
    global lol_game_analysis_df, lol_game_analysis_df_wins
    if there_exists(['hey', 'hi', 'hello']):
        greetings = [f"hey, how can I help you {person_obj.name}", f"hey, what's up? {person_obj.name}",
                     f"I'm listening {person_obj.name}", f"how can I help you? {person_obj.name}",
                     f"hello {person_obj.name}"]
        greet = greetings[random.randint(0, len(greetings) - 1)]
        speak(greet)

    # 2: name
    if there_exists(["what is your name", "what's your name", "tell me your name"]):
        if person_obj.name:
            speak("my name is Sovi")
        else:
            speak("I am Sovi. what's your name?")

    if there_exists(["my name is"]):
        person_name = voice_data.split("is")[-1].strip()
        speak(f"okay, will remember that {person_name}")
        person_obj.setName(person_name)  # remember name in person object

    # 3: greeting
    if there_exists(["how are you", "how are you doing"]):
        speak(f"I'm very well, thanks for asking {person_obj.name}")

    # Start Data Analysis
    if there_exists(["read", "dataset", "analysis"]):
        lol_game_df = pd.read_csv('data/2021_LoL_esports_match_data_from_OraclesElixir_20210404.csv')
        print(lol_game_df.info())

        # print_dataframe = randomforest.RandomForest(lol_game_df, "this is to only print ")
        # print_dataframe.__printData__()

        lol_sample_df = lol_game_df.sample(10)

        lol_selection_df = lol_game_df.loc[:11]

        lol_game_data_null = lol_game_df.isnull().sum()

        lol_game_analysis_df = lol_game_df.drop(
            ['gameid', 'datacompleteness', 'url', 'league', 'year', 'split', 'playoffs',
             'date', 'game', 'patch', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5', 'doublekills',
             'triplekills', 'quadrakills', 'pentakills', 'elementaldrakes',
             'opp_elementaldrakes',
             'infernals', 'mountains', 'clouds', 'oceans', 'dragons (type unknown)',
             'elders',
             'opp_elders', 'opp_heralds', 'barons', 'opp_barons',
             'opp_dragons', 'dragons', 'opp_towers', 'towers', 'heralds',
             'player', 'champion', 'damageshare', 'earnedgoldshare', 'total cs',
             'firstbloodkill',
             'firstbloodassist', 'firstbloodvictim'], axis=1)

        print("\nTotal No of Rows: ", len(lol_game_analysis_df))
        print(lol_game_analysis_df.shape)

        print("\n Use One of the following Commands")
        print("\n1. Show me team stats")
        print("2. Team with Maximum Participation")
        print("3. Team with Maximum Wins")

    # Show Team Stats
    if there_exists(["team", "steam", "dheeme"]):
        lol_game_analysis_df = lol_game_analysis_df.loc[lol_game_analysis_df["playerid"] > 99].drop_duplicates()
        print("No of Rows with only team stats: ", len(lol_game_analysis_df))
        print(lol_game_analysis_df.shape)

        lol_game_analysis_df = lol_game_analysis_df.loc[lol_game_analysis_df["gamelength"] > 99].drop_duplicates()
        lol_game_analysis_df = lol_game_analysis_df.loc[
            lol_game_analysis_df["team"] != 'unknown team'].drop_duplicates()

        print("No of Rows with a known team name: ", len(lol_game_analysis_df))
        print(lol_game_analysis_df.shape)
        display(lol_game_analysis_df)



    # Draw Heatmap
    if there_exists(["heatmap", "heat map"]):
        print("Correlation Heatmap : ", '\n')

        correlationMatrix = lol_game_analysis_df.corr()

        cmap = sns.diverging_palette(10, 150, as_cmap=True)
        correlationMatrix.style.background_gradient(cmap, axis=1) \
            .set_precision(2)
        display(cmap)

    # Maximum Participation
    if there_exists(["maximum", "participation"]):
        print("Total No of Teams: ", len(lol_game_analysis_df['team'].unique()), '\n')
        print("Top 20 Teams with Maximum Participation: ", '\n')

        lol_game_analysis_df_team_count = lol_game_analysis_df['team'].value_counts()[:20]
        print(lol_game_analysis_df_team_count, '\n')

        fig, ax1 = plt.subplots(figsize=(15, 10))
        lol_game_analysis_df_team_count.plot(kind='bar')
        plt.show()

    # Most Wins
    if there_exists(["most", "wins"]):
        lol_game_analysis_df_wins = lol_game_analysis_df.loc[lol_game_analysis_df["result"] != 0].drop_duplicates()
        print("No of Rows with a winning team: ", len(lol_game_analysis_df_wins))
        print(lol_game_analysis_df_wins.shape, '\n')

        print("Top 20 Teams with Maximum Wins: ", '\n')
        lol_game_analysis_df_wins_count = lol_game_analysis_df_wins['team'].value_counts()[:20]
        # myList = lol_game_analysis_df_wins['team'].value_counts()[:20].tolist()
        # print(myList)
        # lol_game_analysis_df_wins_count = lol_game_analysis_df_wins.groupby('result')['team'].value_counts()

        print(lol_game_analysis_df_wins_count, '\n')

        fig, ax1 = plt.subplots(figsize=(15, 10))
        lol_game_analysis_df_wins_count.plot(kind='bar')
        plt.show()

        print("\n Use One of the following Commands")
        print("\n1. Game Time Stats")
        print("2. Game Pie Chart")
        print("3. Winning Strategy")
        print("4. Past records")

    # Time Curve
    if there_exists(['time']):
        def plotGameDuration(data):
            plt.figure(figsize=(15, 10))
            Duration_plot = plt.hist(data['gamelength'], bins=200)
            my_x_ticks = np.arange(0, 4000, 250)
            plt.xticks(my_x_ticks)
            plt.xlabel("gamelength (in sec)", fontsize=13)
            plt.ylabel('Frequency', fontsize=13)
            plt.title('Game Duration Plot', fontsize=15)
            plt.show()

        plotGameDuration(lol_game_analysis_df_wins)

    # Winning Possibility
    if there_exists(["pie", "chart"]):
        lol_game_analysis_df_wins_blue = lol_game_analysis_df_wins.loc[
            lol_game_analysis_df_wins["side"] != 'Red'].drop_duplicates()
        lol_game_analysis_df_wins_red = lol_game_analysis_df_wins.loc[
            lol_game_analysis_df_wins["side"] != 'Blue'].drop_duplicates()

        fig, ax1 = plt.subplots(figsize=(15, 10))
        plt.pie((len(lol_game_analysis_df_wins_blue), len(lol_game_analysis_df_wins_red)),
                labels=('Blue Team Wins', 'Red Team Wins'), startangle=90, autopct='%.2f%%')
        plt.axis('equal')
        plt.title('Winning Rate for 2 Teams')
        plt.show()

    # Win Strategies
    if there_exists(["strategy", "team winning"]):
        percentage_probability_firstBlood_count_wins = (len(
            lol_game_analysis_df_wins.loc[lol_game_analysis_df_wins['firstblood'] > 0]) / len(
            lol_game_analysis_df_wins)) * 100
        percentage_probability_firstDragon_count_wins = (len(
            lol_game_analysis_df_wins.loc[lol_game_analysis_df_wins['firstdragon'] > 0]) / len(
            lol_game_analysis_df_wins)) * 100
        percentage_probability_firstHerald_count_wins = (len(
            lol_game_analysis_df_wins.loc[lol_game_analysis_df_wins['firstherald'] > 0]) / len(
            lol_game_analysis_df_wins)) * 100
        percentage_probability_firstBaron_count_wins = (len(
            lol_game_analysis_df_wins.loc[lol_game_analysis_df_wins['firstbaron'] > 0]) / len(
            lol_game_analysis_df_wins)) * 100
        percentage_probability_firstTower_count_wins = (len(
            lol_game_analysis_df_wins.loc[lol_game_analysis_df_wins['firsttower'] > 0]) / len(
            lol_game_analysis_df_wins)) * 100
        percentage_probability_firstMidTower_count_wins = (len(
            lol_game_analysis_df_wins.loc[lol_game_analysis_df_wins['firstmidtower'] > 0]) / len(
            lol_game_analysis_df_wins)) * 100
        percentage_probability_firstToThreeTowers_count_wins = (len(
            lol_game_analysis_df_wins.loc[lol_game_analysis_df_wins['firsttothreetowers'] > 0]) / len(
            lol_game_analysis_df_wins)) * 100

        probabilities = [percentage_probability_firstBlood_count_wins, percentage_probability_firstDragon_count_wins,
                         percentage_probability_firstHerald_count_wins, percentage_probability_firstBaron_count_wins,
                         percentage_probability_firstTower_count_wins, percentage_probability_firstMidTower_count_wins,
                         percentage_probability_firstToThreeTowers_count_wins]

        labels = ['firstblood', 'firstdragon', 'firstherald', 'firstbaron', 'firsttower', 'firstmidtower',
                  'firsttothreetowers']

        y_pos = np.arange(len(labels))
        plt.figure(figsize=(20, 15))
        plt.bar(y_pos, probabilities, align='center', alpha=1)
        plt.xticks(y_pos, labels, fontsize=30, Rotation=15)
        plt.yticks(fontsize=30)
        plt.ylabel('Win Probability(%)', fontsize=30)
        plt.title('Winning Probability when a Team got First', fontsize=40)
        for a, b in zip(y_pos, probabilities):
            plt.text(a, b, '%.3f' % b + '%', ha='center', va='bottom', fontsize=30)
        plt.show()

    # Previous Years Strategies
    if there_exists(["past", "records"]):
        plt.figure(figsize=(20, 15))
        plt.plot([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
                 [61.874, 64.139, 49.965, 60.735, 62.260, 60.276, 61.362, 61.705], 'o-c', label='firstblood')
        plt.plot([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
                 [62.745, 62.922, 50.807, 50.345, 53.286, 64.694, 59.323, 57.758], '*-y', label='firstdragon')
        plt.plot([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
                 [0.000, 0.166, 35.689, 28.733, 47.292, 59.488, 60.048, 59.311], '+-b', label='firstherald')
        plt.plot([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
                 [79.739, 79.856, 62.883, 62.520, 67.174, 80.545, 77.885, 79.047], 'x-g', label='firstbaron')
        plt.plot([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
                 [61.438, 64.970, 51.790, 67.566, 69.881, 70.094, 67.826, 69.809], 'v-k', label='firsttower')
        plt.plot([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
                 [65.905, 71.389, 58.788, 56.423, 60.678, 75.507, 72.703, 72.706], 'v-r', label='firstmidtower')
        plt.plot([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
                 [73.529, 75.872, 61.385, 60.028, 64.953, 79.925, 77.496, 77.304], 'v-m', label='firsttothreetowers')
        plt.legend(fontsize=30)
        plt.title('Win Probability', fontsize=30)
        plt.ylabel('Win Probability(%)', fontsize=30)
        plt.xlabel('Years', fontsize=30)
        plt.show()

    # Stop and Exit
    if there_exists(["exit", "quit", "goodbye"]):
        speak("going offline")
        exit()


time.sleep(1)
person_obj = person()

print("\n Use One of the following Commands")
print("1. What is your name ?")
print("2. How are you doing ?")
print("3. Read Data for Analysis")

lol_game_data = pd.read_csv('data/2021_LoL_esports_match_data_from_OraclesElixir_20210404.csv')
# print(lol_game_df.info())

print_dataframe = randomforest.RandomForest(lol_game_data, "this is to only print ")
print_dataframe.__printData__()

# print("1. What is Your Name ?")
# print("1. What is Your Name ?")


while 1:
    voice_data = record_audio()  # get the voice input
    respond(voice_data)  # respond
