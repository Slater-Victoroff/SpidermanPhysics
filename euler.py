from time import clock, sleep

def euler(inlist, iterfunc, exitfunc=(lambda vals: True),\
          exittime=float('infinity'), visualize=(lambda vals: True),\
          realtime=False, dt=0.01):
    """ This function takes a tuple of initial conditions and a function
    that will output the derivatives of each input, and will perform a
    simulation until the cows come home (or an optional exit time or set
    of exit conditions are met)"""
    start = clock()
    time = 0
    output = [inlist]
    while exitfunc(output[-1]) and time < exittime:
        output.append([x + dxdt*dt for (x,dxdt) in \
                       zip(output[-1],iterfunc(output[-1]))])
        time += dt
        visualize(output[-1])
        if realtime:
            sleep(dt)
    return output

if __name__ == '__main__':
    iterfunc = (lambda x: x)
    inlist = (1,)
    output = euler(inlist, iterfunc, exittime=1, dt=0.001)
    print "e = %f" %output[-1][0]
