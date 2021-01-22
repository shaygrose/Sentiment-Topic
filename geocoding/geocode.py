import pandas as pd
import spacy
import csv
from geopy.geocoders import Nominatim
import json
from geojson import Feature, FeatureCollection, Point
import requests



def spacy_location(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    final_loc = ''
    locations = []

    for ent in doc.ents:
        if (ent.label_ == 'GPE'):
            locations.append(ent.text)

    if len(locations) > 0:
        loc_counts = pd.DataFrame([[x, locations.count(x)]
                                   for x in set(locations)])
        loc_counts.columns = ['location', 'count']
        loc_counts = loc_counts.sort_values(by=['count'], ascending=False)

        final_loc = loc_counts.iloc[0, 0]

    return final_loc


def geocode_articles(df):

    new_df = df
    gl = Nominatim(user_agent='teletubbies')

    for i in range(len(df)):          # Parses through each article in df

        # Find most frequently mentioned location in article text
        text_loc = spacy_location(new_df.loc[i, 'article'])

        if text_loc != '':    # If location string from article text is not empty
            try:
                print(text_loc)
                # Assign article location to this text
                new_df.loc[i, 'location'] = text_loc
                # Find coordinates of location returned by spacy
                loc = gl.geocode(text_loc)
                lat, lng = loc.latitude, loc.longitude

                # Assign coordinates to article
                new_df.loc[i, 'lat'] = lat
                new_df.loc[i, 'lng'] = lng
                print(str([lat, lng]))
            except:
                print("Error with extracting coordinates.")
                new_df.loc[i, 'lat'] = 0.0
                new_df.loc[i, 'lng'] = 0.0
        else:
            # set location to London by default
            new.df.loc[i, 'location'] = 'London'
            new_df.loc[i, 'lat'] = 51.509865
            new_df.loc[i, 'lng'] = -0.118092
    return new_df



def convert_to_json(filename):
    # convert into a proper geojson format for one article per object
    # the name of the csv you read has to be the same as the one you just wrote that has no index and header and the correct columns
    # the attributes in the for url, title, topic etc may require changing depending on which file you're reading
    features = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)[1:]
        for _, title, _, _, _, _, _, _, _, _, predicted_category, _, lat, lng, location in rows:
            if location != '':
                latitude, longitude = map(float, (lat, lng))
                features.append(
                    Feature(
                        geometry=Point((longitude, latitude)),
                        properties={
                            'topic': title,
                            'title': title,
                            'category': predicted_category,
                            'location': location,
                        }
                    )
                )

    # change X to indicate what you're grouping by
    collection = FeatureCollection(features)
    with open("articles_with_location.json", "w") as f:
        f.write('%s' % collection)


def geo(filename):
    data = pd.read_csv(filename)

    # # # Call geocode articles
    new_df = geocode_articles(data)
    new_df.to_csv('../output/geo/bbc_articles_with_location.csv')
    convert_to_json('../output/geo/bbc_articles_with_location.csv')


if __name__ == '__main__':
    geo('../output/topic/articles_with_predicted_categories.csv')
