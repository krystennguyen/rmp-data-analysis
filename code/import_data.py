#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:20:33 2024

@author: krysten

PREPROCESS CSV DATA AND CREATE rmp.db
"""

import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DB_NAME = '../data/preprocessed/rmp.db'
NUM_CSV = '../data/raw/rmpCapstoneNum.csv'
NUM_HEADER = [
    'avg_rating',
    'avg_diff',
    'n_ratings',
    'pepper',
    'prop_take_again',
    'n_ratings_online',
    'male_clf',
    'female_clf'
]

QUAL_CSV = '../data/raw/rmpCapstoneQual.csv'
QUAL_HEADER = ['major', 'university', 'state']

TAGS_CSV = '../data/raw/rmpCapstoneTags.csv'
TAGS_HEADER = [
    "Tough grader",
    "Good feedback",
    "Respected",
    "Lots to read",
    "Participation matters",
    "Don’t skip class or you will not pass",
    "Lots of homework",
    "Inspirational",
    "Pop quizzes!",
    "Accessible",
    "So many papers",
    "Clear grading",
    "Hilarious",
    "Test heavy",
    "Graded by few things",
    "Amazing lectures",
    "Caring",
    "Extra credit",
    "Group projects",
    "Lecture heavy"
]

# todo: make it a function


def clean_data():
    # read in the csv as dataframes
    num_df = pd.read_csv(NUM_CSV, names=NUM_HEADER)  # (89893, 8)
    qual_df = pd.read_csv(QUAL_CSV, names=QUAL_HEADER)  # (89893, 3))
    tags_df = pd.read_csv(TAGS_CSV, names=TAGS_HEADER)  # (89893, 20)

    # temporary merge the 3 tables row by row (concat column-wise)
    merged_df = pd.concat([num_df, qual_df, tags_df], axis=1)  # (89893, 31)

    # delete empty row if nan values qualitative ratings
    df_clean = merged_df.dropna(
        how='all', subset=['major', 'university', 'state'], axis=0)

    # reset row index
    df_clean = df_clean.reset_index(drop=True)  # (70004,31)

    # describe distribution of number of ratings
    df_clean['n_ratings'].describe()

    print('Proportion of entries with only 1 rating:', len(
        df_clean[df_clean['n_ratings'] == 1])/len(df_clean))
    print('Proportion of entries with at least 3 ratings:', len(
        df_clean[df_clean['n_ratings'] >= 3])/len(df_clean))
    print('Proportion of entries with at least 4 ratings:', len(
        df_clean[df_clean['n_ratings'] >= 4])/len(df_clean))
    print('Proportion of entries with at least 5 ratings:', len(
        df_clean[df_clean['n_ratings'] >= 5])/len(df_clean))

    sns.histplot(df_clean['n_ratings'], bins=1200,
                 color="purple", edgecolor="black")
    plt.title(f'Numbers of Ratings Distribution, N={len(df_clean)}')
    # red line at 5
    plt.axvline(x=5, color='red', linestyle='--')
    plt.savefig('../results/figures/num_ratings_dist.png')

    plt.show()

    # cut off
    df_clean_cutoff = df_clean[df_clean['n_ratings'] >= 5]

    plt.figure()
    sns.histplot(df_clean_cutoff['n_ratings'],
                 bins=1200, color="purple", edgecolor="black")
    plt.title(
        f'Numbers of Ratings Distribution after Cutoff at 5, N={len(df_clean_cutoff)}')
    plt.savefig('../results/figures/num_ratings_dist_cutoff.png')
    plt.show()

    return df_clean_cutoff

# Create database


def create_db():
    conn = sqlite3.connect(DB_NAME)
    # drop table if exists
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS num")
    cursor.execute("DROP TABLE IF EXISTS qual")
    cursor.execute("DROP TABLE IF EXISTS tags")
    conn.commit()
    conn.close()

    print(f"Created DB: {conn}")


# Function to execute SQL query
def make_query(query, param=None):
    # Establish connection to db
    conn = sqlite3.connect(DB_NAME)

    # Read and execute query
    cursor = conn.cursor()

    # If passing as parameter (sanitization)
    if param:
        cursor.execute(query, param)
    else:
        # Standard
        cursor.execute(query)

    # Return result
    result = cursor.fetchall()

    if param:
        print("\n See query...\n")
        print(query, param)
    else:
        print("\n See query...\n")
        print(query)

    print("\n See query result...\n")
    print(result)
    conn.close()

    return result

# Function to create the table with RMP data


def create_update_db_table(query):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(query)

    # Commit changes and close the connection -> write
    conn.commit()
    conn.close()

# TODO: EDIT TO FIT TAB;ES


def insert_from_df(df, label):
    # Connect to db, get cursor
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Execute many insertions at once, use ? as a placeholder to insert variables (our data) in each row safely (sanitization)
    if label == "num":
        for index, row in df.iterrows():
            cursor.execute("""
                INSERT INTO num (
                    avg_rating, avg_diff, n_ratings, pepper, prop_take_again,
                    n_ratings_online, male_clf, female_clf
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (row['avg_rating'], row['avg_diff'], row['n_ratings'], row['pepper'],
                  row['prop_take_again'], row['n_ratings_online'], row['male_clf'],
                  row['female_clf']))

    elif label == "qual":
        for index, row in df.iterrows():
            cursor.execute("""
                INSERT INTO qual (
                    major,university,state
                ) VALUES (?, ?, ?)
            """, (row['major'], row['university'], row['state']))

    elif label == "tags":
        cursor.executemany("""
            INSERT INTO tags (
                "Tough grader",
                "Good feedback",
                "Respected",
                "Lots to read",
                "Participation matters",
                "Don’t skip class or you will not pass",
                "Lots of homework",
                "Inspirational",
                "Pop quizzes!",
                "Accessible",
                "So many papers",
                "Clear grading",
                "Hilarious",
                "Test heavy",
                "Graded by few things",
                "Amazing lectures",
                "Caring",
                "Extra credit",
                "Group projects",
                "Lecture heavy"
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
        """, df.values.tolist())
    else:
        raise ValueError("Please put in correct file.")

    conn.commit()
    conn.close()
    print("\n")


# SCRIPT: to import and preprocess data into SQL DB
# Create rmp.db
def preprocess():
    create_db()
    # preprocess data before insert into table
    preprocessed_df = clean_data()
    # divide this preprocess merged df into 3 tables
    num, qual, tags = preprocessed_df.iloc[:,
                                           :8], preprocessed_df.iloc[:, 8:11], preprocessed_df.iloc[:, 11:]
    print(num.head())
    print(qual.head())
    print(tags.head())

    # Create 'num' ratings table
    # 1: Average Rating (the arithmetic mean of all individual quality ratings of this professor)
    # 2: Average Difficulty (the arithmetic mean of all individual difficulty ratings of this professor)
    # 3: Number of ratings (simply the total number of ratings these averages are based on)
    # 4: Received a “pepper”? (Boolean -was this professor judged as “hot” by the students?)
    # 5: The proportion of students that said they would take the class again
    # 6: The number of ratings coming from online classes
    # 7: Male gender (Boolean –1: determined with high confidence that professor is male)
    # 8: Female (Boolean –1: determined with high confidence that professor is female)
    query = """
        CREATE TABLE IF NOT EXISTS num (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        avg_rating FLOAT,
        avg_diff FLOAT,
        n_ratings INTEGER,
        pepper BOOLEAN,
        prop_take_again INTEGER,
        n_ratings_online INTEGER,
        male_clf BOOLEAN,
        female_clf BOOLEAN
        )
        """
    create_update_db_table(query)
    # populate our table
    insert_from_df(num, 'num')

    # Create 'qual' ratings table
    # Column 1: Major/Field
    # Column 2: University
    # Column 3: US State (2 letter abbreviation
    query = """
        CREATE TABLE IF NOT EXISTS qual (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        major TEXT,
        university TEXT,
        state TEXT
        )
        """
    create_update_db_table(query)
    # populate our table
    insert_from_df(qual, 'qual')

    # Create 'tags' table
    query = """
        CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        "Tough grader" INTEGER,
        "Good feedback" INTEGER,
        "Respected" INTEGER,
        "Lots to read" INTEGER,
        "Participation matters" INTEGER,
        "Don’t skip class or you will not pass" INTEGER,
        "Lots of homework" INTEGER,
        "Inspirational" INTEGER,
        "Pop quizzes!" INTEGER,
        "Accessible" INTEGER,
        "So many papers" INTEGER,
        "Clear grading" INTEGER,
        "Hilarious" INTEGER,
        "Test heavy" INTEGER,
        "Graded by few things" INTEGER,
        "Amazing lectures" INTEGER,
        "Caring" INTEGER,
        "Extra credit" INTEGER,
        "Group projects" INTEGER,
        "Lecture heavy" INTEGER
        )
        """
    create_update_db_table(query)
    insert_from_df(tags, 'tags')


if __name__ == '__main__':
    preprocess()
