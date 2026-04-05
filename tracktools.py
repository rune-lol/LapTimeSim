def get_track():
    track = []
    for _ in range(0, 100):
        track.append((1, 0)) # ds (1m), kappa (1/r)
    for _ in range(0, 100):
        track.append((1, 0.01)) # ds (1m), kappa (1/r) (1/100)
    for i in range(0, 100):
        track.append((0.1, 0.01 - i*0.01)) # ds (1m), kappa (1/r) (1/100 - 1/9999999)
    

    return track