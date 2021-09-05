import sys
import math


def compute_distance(x1, y1, x2, y2):
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


class Bus:
    def __init__(self, routeID, x, y, time):
        self.routeID = routeID # route of the bus
        self.x = x # last coordinates that have been registered for the bus
        self.y = y
        self.time = time # last time that has been registered for the bus
        self.covered_distance = 0 # total distance covered by this bus on this route
        self.elapsed_time = 0 # total time elapsed from first stop to last stop
        return

    def update_bus(self, x, y, time):
        self.covered_distance += compute_distance(self.x, self.y, x, y)
        # update the distance covered by the bus
        self.elapsed_time += (time-self.time)
        # update the time elapsed from the first stop to this stop
        self.x = x # update coordinates and time with the new values
        self.y = y
        self.time = time
        return

    def get_elapsed_time(self):
        return self.elapsed_time

    def get_covered_distance(self):
        return self.covered_distance

    def get_routeID(self):
        return self.routeID


def extract_data(line, bus_dict):
    data = line.split(' ')  # .split returns a list of strings, they should be converted to integers
    for i in range(len(data)):
        data[i] = int(data[i].rstrip('\n')) # .rstrip allows deleting the \n from the last word
    if data[0] not in bus_dict.keys(): # if the bus has not been encountered before
        bus_dict[data[0]] = Bus(data[1], data[2], data[3], data[4]) # then create a Bus object and put it into the dict
    else:
        bus_dict[data[0]].update_bus(data[2], data[3], data[4]) #otherwise update its informations
    return


def load_data(file, bus_dict):
    with open(file, 'r') as f:
        for line in f:
            extract_data(line, bus_dict)
    return


def get_avg_speed_on_route(routeID, buses_list):
    distance = 0
    time = 0
    for bus in buses_list:
        if routeID == bus.get_routeID():
            # for the requested route get all distances and times
            distance += bus.get_covered_distance()
            time += bus.get_elapsed_time()
    if distance == 0:
        return 0 # routeID has not been found, return 0 (I could have handled the situation better, but it's ok)
    else:
        return distance/time # return the average speed


if __name__ == '__main__':
    if len(sys.argv) != 4:
        exit(1)
    bus_dict = {}
    # I will use a dictionary with busIDs as keys and Bus objects as values
    load_data(sys.argv[1], bus_dict)
    if sys.argv[2] == '-b':
        try:
            print("%d - Total Distance: %.1f" % (int(sys.argv[3]), bus_dict[int(sys.argv[3])].get_covered_distance()))
        except KeyError:
            print("Incorrect busID")
    elif sys.argv[2] == '-l':
        print("%d - Avg Speed: %f" % (int(sys.argv[3]), get_avg_speed_on_route(int(sys.argv[3]), bus_dict.values())))
    else:
        print("Incorrect command line arguments, please use -b or -l as flag")
