import tensorflow as tf


def main():
    file = open("testInput.txt", 'r')
    save = open("testOutput.txt", 'w')
    for i in file:
        line = i.strip()
        a = line[0]
        b = line[4]
        c = line[8]
        d = line[12]
        e = line[16]
        f = line[20]
main()