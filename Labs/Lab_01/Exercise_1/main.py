import sys

class Competitor:
    def __init__(self, name, surname, nationality, evaluations):
        self.competitor_data = {'name': name,  # string
                                'surname': surname,  # string
                                'nationality': nationality,  # string
                                'evaluations': evaluations}  # list of floats


def load_data(file, competitors_list):
    with open(file, 'r') as f:
        for line in f:
            info_list = line.split(' ')  # for each line split words separated by a space
            name = info_list[0]
            surname = info_list[1]
            nationality = info_list[2]
            evaluations = sorted(list(float((info_list[i + 3]).rstrip('\n')) for i in range(5)))
            # evaluations need to be modified: remove \n, convert to float, create a list and sort it
            # in order to make next operations easier
            competitors_list.append(Competitor(name, surname, nationality, evaluations))
            # then create an object for a competitor and add it to the list


def compute_score(competitors):
    for competitor in competitors:
        competitor.competitor_data['score'] = 0.0
        for i in range(3):
            competitor.competitor_data['score'] += (competitor.competitor_data['evaluations'][i + 1])
            # sum the evaluations in the middle (this is the reason why I already ordered the evaluations,
            # because I need to ignore the lowest and the highest values)


def final_ranking(competitors):
    competitors.sort(key=lambda competitor: competitor.competitor_data['score'], reverse=True)
    # sort the list of competitors by score (descending)
    print("Final ranking:")
    for i in range(3):
        print("%d: %s %s Score: %.1f" % (i + 1,
                                         competitors[i].competitor_data["name"],
                                         competitors[i].competitor_data["surname"],
                                         competitors[i].competitor_data["score"]))


def best_country(competitors):
    # the idea is to create another dict with nationality:score
    countries = {}
    for competitor in competitors:
        if competitor.competitor_data["nationality"] in countries.keys():
            countries[competitor.competitor_data["nationality"]] += competitor.competitor_data["score"]
        else:
            countries[competitor.competitor_data["nationality"]] = competitor.competitor_data["score"]
    print("")
    print("Best country:")
    max_score = max(countries.values())  # find the highest score
    for code in countries.keys():  # iterate on keys
        if countries[code] == max_score:  # if the value associated with the key is equal to the highest value
            print("%s Total score: %.1f" % (code, max_score))  # print informations


if __name__ == '__main__':
    competitors_list = []
    load_data(sys.argv[1], competitors_list)
    compute_score(competitors_list)
    final_ranking(competitors_list)
    best_country(competitors_list)