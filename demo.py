import tensorflow as tf
import numpy as np
import pandas as pd


class Day:

    def __init__(self, nb_family, date, activity1, place1, nb_ppl_now1, activity2, place2, nb_ppl_now2, activity3,
                 place3, nb_ppl_now3, activity4, place4, nb_ppl_now4, activity5, place5, nb_ppl_npw5):
        self.nb_family = nb_family
        self.date = date
        self.activity1 = activity1
        self.place1 = place1
        self.nb_ppl_now1 = nb_ppl_now1
        self.activity2 = activity2
        self.place2 = place2
        self.nb_ppl_now2 = nb_ppl_now2
        self.activity3 = activity3
        self.place3 = place3
        self.nb_ppl_now3 = nb_ppl_now3
        self.activity4 = activity4
        self.place4 = place4
        self.nb_ppl_now4 = nb_ppl_now4
        self.activity5 = activity5
        self.place5 = place5
        self.nb_ppl_npw5 = nb_ppl_npw5


listdays = []


def loaddata():
    dataframe = pd.read_csv("/home/benarousfarouk/Desktop/IA/implementation/AnomalyDetectionIOT/data.csv")

    dataframe = dataframe.drop(columns='Horodateur')

    inputX = dataframe.loc[:,
             ['nb_family', 'date', 'activity1', 'place1', 'nb_ppl_now1', 'activity2', 'place2', 'nb_ppl_now2',
              'activity3', 'place3', 'nb_ppl_now3', 'activity4', 'place4', 'nb_ppl_now4', 'activity5', 'place5',
              'nb_ppl_npw5']].values

    inputY = dataframe.loc[:, ["Target"]].values

    for i, j in dataframe.iterrows():
        a1 = "0"
        a2 = "0"
        a3 = "0"
        a4 = "0"
        a5 = "0"
        p1 = "0"
        p2 = "0"
        p3 = "0"
        p4 = "0"
        p5 = "0"
        nb_family = "0"
        date = "0"
        nb_ppl_now1 = "0"
        nb_ppl_now2 = "0"
        nb_ppl_now3 = "0"
        nb_ppl_now4 = "0"
        nb_ppl_now5 = "0"
        #####################################"
        if j[2] == "manger":
            a1 = "0"
        elif j[2] == "cuisinier":
            a1 = "1"
        elif j[2] == "regarder la télévision":
            a1 = "2"
        elif j[2] == "étudier":
            a1 = "3"
        elif j[2] == "dormir":
            a1 = "4"
        elif j[2] == "prendre une douche":
            a1 = "5"
        elif j[2] == "faire du sport":
            a1 = "6"
        elif j[2] == "travailler":
            a1 = "7"
        else:
            a1 = "8"
        #####
        if j[5] == "manger":
            a2 = "0"
        elif j[5] == "cuisinier":
            a2 = "1"
        elif j[5] == "regarder la télévision":
            a2 = "2"
        elif j[5] == "étudier":
            a2 = "3"
        elif j[5] == "dormir":
            a2 = "4"
        elif j[5] == "prendre une douche":
            a2 = "5"
        elif j[5] == "faire du sport":
            a2 = "6"
        elif j[5] == "travailler":
            a2 = "7"
        else:
            a2 = "8"
        #####
        if j[8] == "manger":
            a3 = "0"
        elif j[8] == "cuisinier":
            a3 = "1"
        elif j[8] == "regarder la télévision":
            a3 = "2"
        elif j[8] == "étudier":
            a3 = "3"
        elif j[8] == "dormir":
            a3 = "4"
        elif j[8] == "prendre une douche":
            a3 = "5"
        elif j[8] == "faire du sport":
            a3 = "6"
        elif j[8] == "travailler":
            a3 = "7"
        else:
            a3 = "8"
        ########
        if j[11] == "manger":
            a4 = "0"
        elif j[11] == "cuisinier":
            a4 = "1"
        elif j[11] == "regarder la télévision":
            a4 = "2"
        elif j[11] == "étudier":
            a4 = "3"
        elif j[11] == "dormir":
            a4 = "4"
        elif j[11] == "prendre une douche":
            a4 = "5"
        elif j[11] == "faire du sport":
            a4 = "6"
        elif j[11] == "travailler":
            a4 = "7"
        else:
            a4 = "8"
        ########
        if j[14] == "manger":
            a5 = "0"
        elif j[14] == "cuisinier":
            a5 = "1"
        elif j[14] == "regarder la télévision":
            a5 = "2"
        elif j[14] == "étudier":
            a5 = "3"
        elif j[14] == "dormir":
            a5 = "4"
        elif j[14] == "prendre une douche":
            a5 = "5"
        elif j[14] == "faire du sport":
            a5 = "6"
        elif j[14] == "travailler":
            a5 = "7"
        else :
            a5 = "8"
        ######################################

        if j[3] == "salon":
            p1 = "0"
        elif j[3] == "cuisine":
            p1 = "1"
        elif j[3] == "chambre à coucher":
            p1 = "2"
        elif j[3] == "bureau":
            p1 = "3"
        elif j[3] == "salle de bain":
            p1 = "4"
        elif j[3] == "à l'extérieur":
            p1 = "5"
        else:
            p1 = "6"
        ######
        if j[6] == "salon":
            p2 = "0"
        elif j[6] == "cuisine":
            p2 = "1"
        elif j[6] == "chambre à coucher":
            p2 = "2"
        elif j[6] == "bureau":
            p2 = "3"
        elif j[6] == "salle de bain":
            p2 = "4"
        elif j[6] == "à l'extérieur":
            p2 = "5"
        else:
            p2 = "6"
        ######
        if j[9] == "salon":
            p3 = "0"
        elif j[9] == "cuisine":
            p3 = "1"
        elif j[9] == "chambre à coucher":
            p3 = "2"
        elif j[9] == "bureau":
            p3 = "3"
        elif j[9] == "salle de bain":
            p3 = "4"
        elif j[9] == "à l'extérieur":
            p3 = "5"
        else:
            p3 = "6"
        ######
        if j[12] == "salon":
            p4 = "0"
        elif j[12] == "cuisine":
            p4 = "1"
        elif j[12] == "chambre à coucher":
            p4 = "2"
        elif j[12] == "bureau":
            p4 = "3"
        elif j[12] == "salle de bain":
            p4 = "4"
        elif j[12] == "à l'extérieur":
            p4 = "5"
        else:
            p4 = "6"
        ######
        if j[15] == "salon":
            p5 = "0"
        elif j[15] == "cuisine":
            p5 = "1"
        elif j[15] == "chambre à coucher":
            p5 = "2"
        elif j[15] == "bureau":
            p5 = "3"
        elif j[15] == "salle de bain":
            p5 = "4"
        elif j[15] == "à l'extérieur":
            p5 = "5"
        else:
            p5 = "6"
        ################################

        day = Day(j[0], j[1], a1, p1, j[4], a2, p2, j[7], a3, p3, j[10], a4, p4, j[13], a5, p5,
                  j[16])
        listdays.append(day)

    df = pd.DataFrame([t.__dict__ for t in listdays])

    return df


if __name__ == '__main__':
    dataframe = loaddata()
    print(dataframe)

