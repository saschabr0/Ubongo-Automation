import precise


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    precise.connect()
    x, y, z = precise.wherexyz()
    print("x:", x)
    print("y:", y)
    print("z:", z)
    precise.enableDIO(33)
    precise.movexyz(400.0, 15.0, -5.0)
    precise.movexyz(350.0, 15.0, -5.0)
    precise.movexyz(400.0, 15.0, -5.0)
    precise.disableDIO(33)
    precise.disconnect()